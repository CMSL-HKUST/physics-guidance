import os
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from jax import lax
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
jax.config.update("jax_enable_x64", True)

# ==================== Parameter settings (identical to mpm_mls_visco_cont.py) ====================
dim = 2

# Dual-modulus system: separate physical material properties from numerical values
E = 1.8e6
nu = 0.4  # Poisson's ratio (dimensionless)
rho = 1.15e3  # Physical density [kg/m³]

mu_0 = E / (2 * (1 + nu))
dt = 3e-4
damping = 0.0001

# Viscoelastic parameters (generalized Maxwell model - Prony series, Abaqus format)
g_prony = [(0.25, 0.5)]
k_prony = [(0.0, 0.1)]

# Compute long-term moduli
g_sum = sum([g for g, tau in g_prony])
k_sum = sum([k for k, tau in k_prony])
mu_inf = mu_0 * (1.0 - g_sum)
K_0 = E / (3.0 * (1 - 2 * nu))
lambda_inf = 2 * mu_inf * nu / (1 - 2*nu)
K_inf = K_0 * (1.0 - k_sum)

# Physical size parameters
LENGTH_SCALE = 0.05  # Physical length scale [m]
STRESS_SCALE = 0.005

# Surface detection parameters
SURFACE_THICKNESS_CELLS = 2

SHAPE_FILENAME = 'diffusion/samples/print/p3/1/1-2/diffusion/samples/x_t_hstry.npy'
PARTICLES_PER_CELL_DIM = 2
grid_padding_cells = 20

# Compression parameters
compression_velocity = 0.1
max_compression_strain = 0.5
gravity = 0

# Self-contact parameters
self_contact_friction = 0.03


# ==================== Helper functions ====================

def quadratic_bspline_weights(fx):
    """Compute quadratic B-spline weights"""
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return w0, w1, w2


def safe_norm(x, axis=-1, keepdims=False, eps=1e-15):
    """Safe norm to avoid NaN gradients at zero"""
    sq_sum = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(sq_sum + eps)


def compute_grid_gradient(grid_m, inv_dx):
    """Compute gradient of the mass field (differentiable version)"""
    padded = jnp.pad(grid_m, 1, mode='edge')
    grad_x = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5 * inv_dx
    grad_y = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5 * inv_dx
    return jnp.stack([grad_x, grad_y], axis=-1)


def update_internal_variables(q_shear_old, q_bulk_old, D):
    """Update internal variables (strictly following mpm_mls_visco_cont.py)"""
    trace_D = jnp.trace(D, axis1=1, axis2=2)
    D_dev = D - (trace_D[:, None, None] / 2.0) * jnp.eye(dim)
    
    q_shear_new = jnp.zeros_like(q_shear_old)
    for i, (g_i, tau_i) in enumerate(g_prony):
        exp_factor = jnp.exp(-dt / tau_i)
        G_i = mu_0 * g_i
        source_shear = 2.0 * D_dev
        q_shear_new = q_shear_new.at[:, i, :, :].set(
            exp_factor * q_shear_old[:, i, :, :] + 
            G_i * tau_i * (1.0 - exp_factor) * source_shear
        )
    
    q_bulk_new = jnp.zeros_like(q_bulk_old)
    for i, (k_i, tau_i) in enumerate(k_prony):
        exp_factor = jnp.exp(-dt / tau_i)
        K_i = K_0 * k_i
        source_bulk = trace_D
        q_bulk_new = q_bulk_new.at[:, i].set(
            exp_factor * q_bulk_old[:, i] + 
            K_i * tau_i * (1.0 - exp_factor) * source_bulk
        )
    
    return q_shear_new, q_bulk_new


def compute_viscoelastic_stress(F, D, q_shear, q_bulk):
    """Compute viscoelastic stress (strictly following mpm_mls_visco_cont.py)"""
    # Compute determinant
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    J_safe = jnp.maximum(J, 1e-9)
    
    # Compute F^{-T}
    F_inv_T = jnp.stack([
        jnp.stack([F[:, 1, 1], -F[:, 1, 0]], axis=-1),
        jnp.stack([-F[:, 0, 1], F[:, 0, 0]], axis=-1)
    ], axis=1) / J_safe[:, None, None]
    
    # Base elastic stress
    P_elastic = mu_inf * (F - F_inv_T) + lambda_inf * jnp.log(J_safe)[:, None, None] * F_inv_T
    
    # Viscous correction
    sigma_shear_visc = jnp.sum(q_shear, axis=1)
    sigma_bulk_visc = jnp.sum(q_bulk, axis=1)
    sigma_visc = sigma_shear_visc + sigma_bulk_visc[:, None, None] * jnp.eye(dim)
    P_visc = J_safe[:, None, None] * jnp.einsum('pij,pjk->pik', sigma_visc, F_inv_T)
    
    P_total = P_elastic + P_visc
    P_total_scaled = P_total * STRESS_SCALE
    
    return P_total_scaled


# ==================== Initialization functions ====================

def init_particles_from_density(voxel_density, dx):
    """Initialize particles from a density field (supports viscoelasticity and partitioned contact)
    
    Strictly follows the logic of init_state_from_matrix in mpm_mls_visco_cont.py,
    but supports density-field input to enable automatic differentiation.
    
    Returns: p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad_mass
    """
    voxel_density = jnp.flipud(voxel_density)
    
    # Select voxels with density (use a small threshold to avoid floating-point errors)
    solid_mask = voxel_density >= 0.5
    solid_indices = jnp.argwhere(solid_mask)  # [n_solid_cells, 2]
    
    # Create sub-particle grid
    sub_spacing = dx / PARTICLES_PER_CELL_DIM
    sub_offsets = jnp.arange(PARTICLES_PER_CELL_DIM) * sub_spacing + sub_spacing * 0.5
    sub_x, sub_y = jnp.meshgrid(sub_offsets, sub_offsets)
    sub_grid = jnp.stack([sub_x.ravel(), sub_y.ravel()], axis=-1)  # [n_sub, 2]
    n_sub = PARTICLES_PER_CELL_DIM ** 2
    
    # Create particles for each solid voxel
    base_positions = solid_indices[:, ::-1] * dx
    p_x = base_positions[:, None, :] + sub_grid[None, :, :]
    p_x = p_x.reshape(-1, 2)
    
    # Center the object in the grid
    p_x_min = jnp.min(p_x, axis=0)
    p_x_max = jnp.max(p_x, axis=0)
    object_center = (p_x_min + p_x_max) / 2.0
    grid_center = jnp.array([0.5, 0.5])
    offset = grid_center - object_center
    p_x = p_x + offset
    
    # Get density for each particle
    cell_densities = voxel_density[solid_indices[:, 0], solid_indices[:, 1]]
    particle_density = jnp.repeat(cell_densities, n_sub)
    
    # Initialize other variables
    n_particles = p_x.shape[0]
    p_v = jnp.zeros((n_particles, dim))
    p_F = jnp.tile(jnp.eye(dim)[None, :, :], (n_particles, 1, 1))
    p_C = jnp.zeros((n_particles, dim, dim))
    
    p_vol_0 = sub_spacing**2
    p_mass = p_vol_0 * rho
    
    # Modulate mass and volume by density
    p_mass = p_mass * particle_density
    p_vol0 = p_vol_0 * particle_density
    
    # Initialize viscoelastic internal variables
    n_prony_shear = len(g_prony)
    n_prony_bulk = len(k_prony)
    p_q_shear = jnp.zeros((n_particles, n_prony_shear, dim, dim))
    p_q_bulk = jnp.zeros((n_particles, n_prony_bulk))
    
    # Initialize particle gradient (for partitioning)
    p_grad_mass = jnp.zeros((n_particles, dim))
    
    return p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad_mass


# ==================== Core MPM functions ====================

@partial(jit, static_argnums=(10, 11))
def mpm_step_arrays(p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad_mass,
                    plate_y, n_grid, inv_dx, ground_y_pos, plate_v):
    """Single-step MLS-MPM update (strictly following the logic of mpm_mls_visco_cont.py)
    
    Uses array-based state, supports viscoelasticity and partitioned contact, and is fully differentiable.
    """
    dx = 1.0 / inv_dx
    
    # 1. Update internal variables (following mpm_mls_visco_cont.py)
    L = p_C
    D = 0.5 * (L + L.transpose((0, 2, 1)))
    new_q_shear, new_q_bulk = update_internal_variables(p_q_shear, p_q_bulk, D)
    
    # 2. Update deformation gradient (following mpm_mls_visco_cont.py)
    F_update_term = jnp.eye(dim) + dt * p_C  # shape: (n_particles, 2, 2)
    new_p_F = jnp.einsum('pij,pjk->pik', F_update_term, p_F)
    
    # 3. Compute mass field and its gradient (following mpm_mls_visco_cont.py)
    grid_m_total = jnp.zeros((n_grid, n_grid))
    base = jnp.floor(p_x * inv_dx - 0.5).astype(jnp.int32)
    fx = p_x * inv_dx - base.astype(jnp.float64)
    
    w0, w1, w2 = quadratic_bspline_weights(fx)
    weights = jnp.stack([w0, w1, w2], axis=1)
    
    for i in range(3):
        for j in range(3):
            weight = weights[:, i, 0] * weights[:, j, 1]
            grid_idx = base + jnp.array([i, j])
            grid_idx = jnp.clip(grid_idx, 0, n_grid - 1)
            grid_m_total = grid_m_total.at[grid_idx[:, 0], grid_idx[:, 1]].add(
                weight * p_mass
            )
    
    grid_grad = compute_grid_gradient(grid_m_total, inv_dx)
    
    # 更新粒子梯度（遵循 mpm_mls_visco_cont.py）
    new_p_grad = jnp.zeros_like(p_grad_mass)
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx = jnp.clip(grid_idx, 0, n_grid - 1)
            weight = weights[:, i, 0] * weights[:, j, 1]
            g_g = grid_grad[grid_idx[:, 0], grid_idx[:, 1]]
            new_p_grad = new_p_grad + weight[:, None] * g_g
    
            # 4. Partitioned P2G (following the partition strategy in mpm_mls_visco_cont.py)
    grid_m_stack = jnp.zeros((2, n_grid, n_grid))
    grid_p_stack = jnp.zeros((2, n_grid, n_grid, dim))
    
    # Compute gradient threshold: separate interior and surface particles (following mpm_mls_visco_cont.py)
    grad_norm_particles = jnp.linalg.norm(new_p_grad, axis=-1)
    grad_threshold_partition = 0.3 * jnp.max(grad_norm_particles)
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx = jnp.clip(grid_idx, 0, n_grid - 1)
            weight = weights[:, i, 0] * weights[:, j, 1]
            dpos = (offset.astype(jnp.float64) - fx) * dx
            g_grad = grid_grad[grid_idx[:, 0], grid_idx[:, 1]]
            dot_prod = jnp.einsum('pi,pi->p', new_p_grad, g_grad)
            
            # Partition strategy (following mpm_mls_visco_cont.py):
            # 1. Interior particles (small gradient) → all in partition 0
            # 2. Surface particles (large gradient) → partitioned by sign of dot product
            is_surface_particle = grad_norm_particles > grad_threshold_partition
            mask_0 = jnp.where(is_surface_particle, (dot_prod >= 0), True)
            
            v_at_grid = p_v + jnp.einsum('pij,pj->pi', p_C, dpos)
            momentum = p_mass[:, None] * v_at_grid
            mass_contrib = weight * p_mass
            mom_contrib = weight[:, None] * momentum
            
            grid_m_stack = grid_m_stack.at[0, grid_idx[:, 0], grid_idx[:, 1]].add(
                jnp.where(mask_0, mass_contrib, 0.0)
            )
            grid_p_stack = grid_p_stack.at[0, grid_idx[:, 0], grid_idx[:, 1]].add(
                jnp.where(mask_0[:, None], mom_contrib, 0.0)
            )
            grid_m_stack = grid_m_stack.at[1, grid_idx[:, 0], grid_idx[:, 1]].add(
                jnp.where(~mask_0, mass_contrib, 0.0)
            )
            grid_p_stack = grid_p_stack.at[1, grid_idx[:, 0], grid_idx[:, 1]].add(
                jnp.where(~mask_0[:, None], mom_contrib, 0.0)
            )
    
    # 5. Compute velocities (partitioned version, following mpm_mls_visco_cont.py)
    m0 = grid_m_stack[0]
    v0 = jnp.where(m0[:, :, None] > 1e-15, grid_p_stack[0] / (m0[:, :, None] + 1e-15), 0.0)
    m1 = grid_m_stack[1]
    v1 = jnp.where(m1[:, :, None] > 1e-15, grid_p_stack[1] / (m1[:, :, None] + 1e-15), 0.0)
    
    # 6. Compute internal force (following mpm_mls_visco_cont.py)
    P = compute_viscoelastic_stress(new_p_F, D, new_q_shear, new_q_bulk)
    F_T = new_p_F.transpose((0, 2, 1))
    kirchhoff_stress = jnp.einsum('pij,pkj->pik', P, new_p_F)
    
    D_inv = 4.0 / (dx * dx)
    stress_term = -D_inv * p_vol0[:, None, None] * kirchhoff_stress
    
    grid_f_total = jnp.zeros((n_grid, n_grid, dim))
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx = jnp.clip(grid_idx, 0, n_grid - 1)
            weight = weights[:, i, 0] * weights[:, j, 1]
            dpos = (offset.astype(jnp.float64) - fx) * dx
            force_contrib = weight[:, None] * jnp.einsum('pij,pj->pi', stress_term, dpos)
            grid_f_total = grid_f_total.at[grid_idx[:, 0], grid_idx[:, 1]].add(force_contrib)
    
            # 7. Update velocities (distribute force proportionally to mass, following mpm_mls_visco_cont.py)
    m_total = m0 + m1 + 1e-15
    f0 = grid_f_total * (m0 / m_total)[:, :, None]
    f1 = grid_f_total * (m1 / m_total)[:, :, None]
    
    v0_updated = v0 + dt * (f0 / (m0[:, :, None] + 1e-15) + jnp.array([0.0, gravity]))
    v1_updated = v1 + dt * (f1 / (m1[:, :, None] + 1e-15) + jnp.array([0.0, gravity]))
    
    v0_damped = v0_updated * (1.0 - damping)
    v1_damped = v1_updated * (1.0 - damping)
    
    # 8. Boundary conditions (following mpm_mls_visco_cont.py)
    y_coords = jnp.arange(n_grid) * dx
    bc_thickness = 1.0 * dx
    
    # Ground
    ground_mask = (y_coords <= ground_y_pos + bc_thickness)
    v0_damped = jnp.where(ground_mask[None, :, None], jnp.zeros_like(v0_damped), v0_damped)
    v1_damped = jnp.where(ground_mask[None, :, None], jnp.zeros_like(v1_damped), v1_damped)
    
    # Compression plate
    dist_to_plate = plate_y - y_coords
    t = dist_to_plate / bc_thickness
    t_clipped = jnp.clip(t, 0.0, 1.0)
    alpha = jnp.where(
        dist_to_plate <= 0, 1.0,
        jnp.where(dist_to_plate < bc_thickness, 0.5 * (1.0 + jnp.cos(jnp.pi * t_clipped)), 0.0)
    )
    
    target_v_plate = jnp.array([0.0, plate_v])
    alpha_grid = alpha[None, :, None]
    v0_damped = (1.0 - alpha_grid) * v0_damped + alpha_grid * target_v_plate
    v1_damped = (1.0 - alpha_grid) * v1_damped + alpha_grid * target_v_plate
    
    # Left and right walls
    v0_damped = v0_damped.at[0, :, 0].set(jnp.maximum(v0_damped[0, :, 0], 0.0))
    v0_damped = v0_damped.at[n_grid-1, :, 0].set(jnp.minimum(v0_damped[n_grid-1, :, 0], 0.0))
    v1_damped = v1_damped.at[0, :, 0].set(jnp.maximum(v1_damped[0, :, 0], 0.0))
    v1_damped = v1_damped.at[n_grid-1, :, 0].set(jnp.minimum(v1_damped[n_grid-1, :, 0], 0.0))

    # 9. Self-contact handling (following mpm_mls_visco_cont.py, using a differentiable vectorized version)
    grad_norm = safe_norm(grid_grad, axis=-1)
    normal = grid_grad / (grad_norm[:, :, None] + 1e-15)
    
    # Compute thresholds for contact detection (following mpm_mls_visco_cont.py)
    total_mass_sum = jnp.sum(grid_m_total)
    nonzero_count = jnp.sum(grid_m_total > 0)
    avg_mass = total_mass_sum / jnp.maximum(nonzero_count, 1.0)
    mass_threshold = 0.01 * avg_mass
    grad_threshold = 0.3 * jnp.max(grad_norm)
    
    # Vectorized soft-contact detection and resolution (avoids nested vmap for performance)
    mass_sharpness = 1000.0
    grad_sharpness = 1000.0
    approach_sharpness = 1000.0
    
    # Condition 1: both partitions have significant mass
    m0_active = jax.nn.sigmoid((m0 - mass_threshold) * mass_sharpness)
    m1_active = jax.nn.sigmoid((m1 - mass_threshold) * mass_sharpness)
    
    # Condition 2: gradient is sufficiently large
    grad_active = jax.nn.sigmoid((grad_norm - grad_threshold) * grad_sharpness)
    
    # Condition 3: relative motion is approaching
    v_rel = v0_damped - v1_damped
    vn = jnp.sum(v_rel * normal, axis=-1, keepdims=True)
    approach_active = jax.nn.sigmoid(-vn * approach_sharpness)
    
    # Combined contact activation probability
    contact_active_prob = m0_active[:, :, None] * m1_active[:, :, None] * grad_active[:, :, None] * approach_active
    
    # Compute post-contact velocities (vectorized version)
    mu = self_contact_friction
    m_sum = m0[:, :, None] + m1[:, :, None] + 1e-15
    v_cm = (m0[:, :, None] * v0_damped + m1[:, :, None] * v1_damped) / m_sum
    
    v1_n = jnp.sum(v0_damped * normal, axis=-1, keepdims=True) * normal
    v1_t = v0_damped - v1_n
    v2_n = jnp.sum(v1_damped * normal, axis=-1, keepdims=True) * normal
    v2_t = v1_damped - v2_n
    v_cm_n = jnp.sum(v_cm * normal, axis=-1, keepdims=True) * normal
    
    vt_rel = v1_t - v2_t
    vt_rel_norm = safe_norm(vt_rel, axis=-1, keepdims=True)
    
    rel_v_norm_safe = safe_norm(v_cm_n - v1_n, axis=-1, keepdims=True)
    normal_impulse = m0[:, :, None] * rel_v_norm_safe
    scale = jnp.minimum(1.0, mu * normal_impulse / (m0[:, :, None] * vt_rel_norm + 1e-15))
    
    v0_contact = v_cm_n + v1_t - vt_rel * scale * 0.5
    v1_contact = v_cm_n + v2_t + vt_rel * scale * 0.5
    
    # Soft blending
    v0_final = (1.0 - contact_active_prob) * v0_damped + contact_active_prob * v0_contact
    v1_final = (1.0 - contact_active_prob) * v1_damped + contact_active_prob * v1_contact
    
    # Contact mask (for visualization)
    contact_mask = contact_active_prob[:, :, 0]
    
    # Compute reaction force (following mpm_mls_visco_cont.py)
    grid_f_y = grid_f_total[:, :, 1]
    alpha_2d = alpha[None, :]
    reaction_force = -jnp.sum(alpha_2d * grid_f_y)
    
    # 10. G2P transfer (following the logic of mpm_mls_visco_cont.py)
    new_p_v = jnp.zeros_like(p_v)
    p_B = jnp.zeros_like(p_C)
    p_D = jnp.zeros_like(p_C)
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx = jnp.clip(grid_idx, 0, n_grid - 1)
            weight = weights[:, i, 0] * weights[:, j, 1]
            g_grad = grid_grad[grid_idx[:, 0], grid_idx[:, 1]]
            
            dot_prod = vmap(jnp.dot)(new_p_grad, g_grad)
            v_node_0 = v0_final[grid_idx[:, 0], grid_idx[:, 1]]
            v_node_1 = v1_final[grid_idx[:, 0], grid_idx[:, 1]]
            
            # Partition strategy consistent with P2G (following mpm_mls_visco_cont.py)
            is_surface_particle = grad_norm_particles > grad_threshold_partition
            mask = jnp.where(is_surface_particle, (dot_prod >= 0), True)[:, None]
            v_on_grid = jnp.where(mask, v_node_0, v_node_1)
            
            dpos = (offset.astype(jnp.float64) - fx) * dx
            new_p_v = new_p_v + weight[:, None] * v_on_grid
            # Use einsum to improve performance
            p_B = p_B + weight[:, None, None] * jnp.einsum('pi,pj->pij', v_on_grid, dpos)
            p_D = p_D + weight[:, None, None] * jnp.einsum('pi,pj->pij', dpos, dpos)
    
    p_D = p_D + jnp.eye(dim) * 1e-9
    p_D_inv = vmap(jnp.linalg.inv)(p_D)
    new_p_C = jnp.einsum('pij,pjk->pik', p_B, p_D_inv)
    new_p_x = p_x + dt * new_p_v
    
    # 11. Particle-level boundary constraints (hard boundaries following mpm_mls_visco_cont.py)
    # Ground constraint
    below_ground = new_p_x[:, 1] < ground_y_pos
    new_p_x = jnp.where(
        below_ground[:, None],
        jnp.stack([new_p_x[:, 0], jnp.full_like(new_p_x[:, 1], ground_y_pos)], axis=1),
        new_p_x
    )
    new_p_v = jnp.where(below_ground[:, None], jnp.zeros_like(new_p_v), new_p_v)
    
    # Compression-plate constraint
    above_plate = new_p_x[:, 1] > plate_y
    new_p_x = jnp.where(
        above_plate[:, None],
        jnp.stack([new_p_x[:, 0], jnp.full_like(new_p_x[:, 1], plate_y)], axis=1),
        new_p_x
    )
    new_p_v = jnp.where(
        above_plate[:, None] & (new_p_v[:, 1:2] > 0),
        jnp.concatenate([new_p_v[:, 0:1], jnp.zeros_like(new_p_v[:, 1:2])], axis=-1),
        new_p_v
    )
    
    return new_p_x, new_p_v, new_p_F, new_p_C, new_q_shear, new_q_bulk, new_p_grad, reaction_force, contact_mask


# ==================== Simulation runner functions ====================

def run_simulation(voxel_density, n_steps, n_grid, inv_dx, initial_y_top, ground_y_pos):
    """Run simulation (fully differentiable version).
    
    Uses checkpointing to avoid memory explosion and returns the final reaction force.
    """
    dx = 1.0 / inv_dx
    
    # Initialization
    p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad = \
        init_particles_from_density(voxel_density, dx)
    
    plate_v = -compression_velocity
    
    @jax.checkpoint
    def checkpointed_body_fn(carry, i):
        p_x, p_v, p_F, p_C, p_q_shear, p_q_bulk, p_grad, p_mass, p_vol0 = carry
        current_time = i * dt
        plate_y = initial_y_top - compression_velocity * current_time
        
        new_p_x, new_p_v, new_p_F, new_p_C, new_q_shear, new_q_bulk, new_p_grad, reaction_force, contact_mask = \
            mpm_step_arrays(
                p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad,
                plate_y, n_grid, inv_dx, ground_y_pos, plate_v
            )
        
        return (new_p_x, new_p_v, new_p_F, new_p_C, new_q_shear, new_q_bulk, new_p_grad, p_mass, p_vol0), reaction_force
    
    carry = (p_x, p_v, p_F, p_C, p_q_shear, p_q_bulk, p_grad, p_mass, p_vol0)
    
    # Use scan to collect reaction forces
    final_carry, all_forces = lax.scan(checkpointed_body_fn, carry, jnp.arange(n_steps))
    
    # Take reaction force at the last step
    reaction_force = all_forces[-1]
    
    # Convert to physical units
    force_N = jnp.abs(reaction_force) * 0.6
    
    # Unpack state for return
    p_x, p_v, p_F, p_C, p_q_shear, p_q_bulk, p_grad, _, _ = final_carry
    return force_N, (p_x, p_v, p_F, p_C)


def run_simulation_with_curve(voxel_density, n_steps, n_grid, inv_dx, 
                              initial_y_top, ground_y_pos, checkpoint_every=100):
    """Run MPM simulation and record the force–displacement curve (fully differentiable version).
    
    Args:
        voxel_density: Density field [ny, nx]
        n_steps: Total number of steps
        n_grid: Grid resolution
        inv_dx: 1/dx
        initial_y_top: Initial top position
        ground_y_pos: Ground position
        checkpoint_every: Checkpoint interval
    
    Returns:
        force_curve: Force curve [n_segments,] in [N]
        displacement_curve: Displacement curve [n_segments,] in [mm]
    """
    if n_steps % checkpoint_every != 0:
        raise ValueError(f"n_steps ({n_steps}) 必须能被 checkpoint_every ({checkpoint_every}) 整除")
    
    dx = 1.0 / inv_dx
    
    # Initialization
    p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad = \
        init_particles_from_density(voxel_density, dx)
    
    plate_v = -compression_velocity
    n_segments = n_steps // checkpoint_every
    
    # Define a single checkpoint segment
    @jax.checkpoint
    def run_one_segment(carry, seg_idx):
        p_x, p_v, p_F, p_C, p_q_shear, p_q_bulk, p_grad = carry
        
        segment_start = seg_idx * checkpoint_every
        
        def body_fn(inner_state, i):
            px, pv, pF, pC, pq_shear, pq_bulk, pg = inner_state
            step_i = segment_start + i
            current_time = step_i * dt
            plate_y = initial_y_top - compression_velocity * current_time
            
            new_px, new_pv, new_pF, new_pC, new_q_shear, new_q_bulk, new_pg, reaction_force, _ = \
                mpm_step_arrays(
                    px, pv, pF, pC, p_mass, p_vol0, pq_shear, pq_bulk, pg,
                    plate_y, n_grid, inv_dx, ground_y_pos, plate_v
                )
            return (new_px, new_pv, new_pF, new_pC, new_q_shear, new_q_bulk, new_pg), reaction_force
        
        # Run checkpoint_every steps
        final_state, reaction_forces = lax.scan(
            body_fn,
            (p_x, p_v, p_F, p_C, p_q_shear, p_q_bulk, p_grad),
            jnp.arange(checkpoint_every)
        )
        
        return final_state, reaction_forces[-1]
    
    # Run all segments
    initial_state = (p_x, p_v, p_F, p_C, p_q_shear, p_q_bulk, p_grad)
    _, all_reaction_forces = lax.scan(run_one_segment, initial_state, jnp.arange(n_segments))
    
    # Convert to physical units
    force_curve = jnp.abs(all_reaction_forces) * 0.6  # [N]
    
    # Compute displacement curve
    record_steps = jnp.arange(1, n_segments + 1) * checkpoint_every
    record_times = record_steps * dt
    displacement_curve = (compression_velocity * record_times) * LENGTH_SCALE * 1000  # [mm]
    
    return force_curve, displacement_curve


def run_simulation_with_history(voxel_density, n_steps, n_grid, inv_dx, 
                                initial_y_top, ground_y_pos, save_every=100):
    """Run simulation and save history (for visualization).
    
    Note: this function is for visualization only and is not used in automatic differentiation.
    """
    dx = 1.0 / inv_dx
    
    # Initialization
    p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad = \
        init_particles_from_density(voxel_density, dx)
    
    plate_v = -compression_velocity
    
    # History buffers
    positions_history = []
    stress_history = []
    displacement_history = []
    force_history = []
    partition_history = []
    contact_nodes_history = []
    
    for i in range(n_steps):
        current_time = i * dt
        plate_y = initial_y_top - compression_velocity * current_time
        
        p_x, p_v, p_F, p_C, p_q_shear, p_q_bulk, p_grad, reaction_force, contact_mask = \
            mpm_step_arrays(
                p_x, p_v, p_F, p_C, p_mass, p_vol0, p_q_shear, p_q_bulk, p_grad,
                plate_y, n_grid, inv_dx, ground_y_pos, plate_v
            )
        
        if (i + 1) % save_every == 0:
            # Compute stress
            L = p_C
            D = 0.5 * (L + L.transpose((0, 2, 1)))
            P = compute_viscoelastic_stress(p_F, D, p_q_shear, p_q_bulk)
            cauchy_stress = (1.0 / jnp.maximum(jnp.abs(vmap(jnp.linalg.det)(p_F)), 1e-9))[:, None, None] * \
                           jnp.einsum('pij,pkj->pik', P, p_F)
            s_xx, s_yy, s_xy = cauchy_stress[:, 0, 0], cauchy_stress[:, 1, 1], cauchy_stress[:, 0, 1]
            von_mises = jnp.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2)
            
            # Physical quantities
            plate_displacement = compression_velocity * current_time
            plate_displacement_mm = plate_displacement * LENGTH_SCALE * 1000
            force_N = jnp.abs(reaction_force) * 0.6
            
            # Compute partition (directly use p_grad to avoid recomputing)
            grad_norm_particles = safe_norm(p_grad, axis=-1)
            grad_threshold_partition = 0.3 * jnp.max(grad_norm_particles)
            is_surface_particle = grad_norm_particles > grad_threshold_partition
            
            # Simple partition (for visualization) - directly use current particle gradients
            partition = jnp.where(is_surface_particle, 1, 0)
            
            # Extract contact nodes
            contact_mask_np = np.array(contact_mask > 0.5)
            indices = np.argwhere(contact_mask_np)
            contact_nodes_pos = indices.astype(float) * dx
            
            n_contact = len(contact_nodes_pos)
            
            print(f"Step: {i+1:>6d}/{n_steps} | "
                  f"Time: {current_time:.4f}s | "
                  f"Displacement: {plate_displacement_mm:6.2f} mm | "
                  f"Force: {force_N:8.2f} N | "
                  f"Contact nodes: {n_contact}")
            
            positions_history.append(np.array(p_x))
            stress_history.append(np.array(von_mises))
            displacement_history.append(plate_displacement_mm)
            force_history.append(float(force_N))
            partition_history.append(np.array(partition))
            contact_nodes_history.append(contact_nodes_pos)
    
    return (positions_history, stress_history, displacement_history, force_history,
            partition_history, contact_nodes_history)


# ==================== Main function ====================

def main():
    """Main entry: run simulation and visualization."""
    
    # Load experimental data
    exp_file = 'experiment/20260105/thick_20mm/2.txt'
    exp_data = pd.read_csv(exp_file, sep='\t', skiprows=1)
    exp_displacement = exp_data.iloc[:, 1].values
    exp_force = exp_data.iloc[:, 0].values
    print(f"Loaded experimental data: {len(exp_force)} points")
    print(f"  Displacement range: {exp_displacement.min():.2f} - {exp_displacement.max():.2f} mm")
    print(f"  Force range: {exp_force.min():.2f} - {exp_force.max():.2f} N")
    
    # Load voxel shape
    voxel_2d = np.round(np.load(SHAPE_FILENAME)).astype(int)
    voxel_2d = voxel_2d[19].squeeze()
    shape_matrix = jnp.array(voxel_2d, dtype=jnp.float64)
    
    n_grid = max(shape_matrix.shape) + 2 * grid_padding_cells
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    
    print(f"\nGrid resolution: {n_grid}x{n_grid}")
    print(f"Cell size: {dx:.4e} (normalized), {dx * LENGTH_SCALE * 1000:.4f} mm (physical)")
    print(f"Physical length scale: {LENGTH_SCALE * 1000:.1f} mm = {LENGTH_SCALE * 100:.1f} cm")
    print(f"\nViscoelastic parameters (Generalized Maxwell - Prony series):")
    print(f"  - Shear Prony: g_i, τ_i = {g_prony}")
    print(f"  - Bulk Prony: k_i, τ_i = {k_prony}")
    
    # Compute initial geometric parameters
    voxel_density_init = shape_matrix
    temp_state = init_particles_from_density(voxel_density_init, dx)
    p_x_init = temp_state[0]
    
    initial_y_min = jnp.min(p_x_init[:, 1])
    initial_y_top = jnp.max(p_x_init[:, 1])
    initial_height = initial_y_top - initial_y_min
    ground_y_pos = initial_y_min
    
    compression_distance = max_compression_strain * initial_height
    compression_time = compression_distance / compression_velocity
    n_steps = int(compression_time / dt)
    
    print(f"\nTotal particles: {p_x_init.shape[0]}")
    print(f"Initial height: {initial_height:.4f} (normalized), {initial_height * LENGTH_SCALE * 1000:.2f} mm (physical)")
    print(f"Compressing for {n_steps} steps...")
    
    # Run simulation
    save_every = 100
    start_time = time.time()
    
    history = run_simulation_with_history(
        voxel_density_init, n_steps, n_grid, inv_dx,
        initial_y_top, ground_y_pos, save_every=save_every
    )
    
    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f} seconds!")
    
    positions_history, stress_history, displacement_history, force_history, \
        partition_history, contact_nodes_history = history
    


    record = np.stack([force_history, displacement_history], axis=1)
    # np.save('experiment/20260117/simu_4-8.npy', record)
    np.save('experiment/1-2/simu_1-2-020.npy', record)


    
    # Visualization
    print("\nCreating animation...")
    
    fig = plt.figure(figsize=(22, 7), dpi=120)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Stress
    ax_partition = fig.add_subplot(gs[0, 1])  # Partition
    ax2 = fig.add_subplot(gs[0, 2])  # Force–displacement curve
    
    # Left: stress
    scatter = ax1.scatter([], [], s=1, c=[], cmap='viridis')
    top_wall_patch = patches.Rectangle((0, initial_y_top), 1, 0.01, fc='grey')
    ax1.add_patch(top_wall_patch)
    ground_wall_patch = patches.Rectangle((0, ground_y_pos - 0.01), 1, 0.01, fc='grey')
    ax1.add_patch(ground_wall_patch)
    cbar1 = fig.colorbar(scatter, ax=ax1)
    cbar1.set_label("Von Mises Stress (Pa)")
    
    # Middle: partition and contact
    scatter_p0 = ax_partition.scatter([], [], s=2, c='blue', alpha=0.6, label='Partition 0')
    scatter_p1 = ax_partition.scatter([], [], s=2, c='red', alpha=0.6, label='Partition 1')
    scatter_contact = ax_partition.scatter([], [], s=50, c='yellow', marker='*', 
                                           edgecolors='black', linewidths=0.5,
                                           label='Contact Nodes', zorder=10)
    top_wall_patch2 = patches.Rectangle((0, initial_y_top), 1, 0.01, fc='grey')
    ax_partition.add_patch(top_wall_patch2)
    ground_wall_patch2 = patches.Rectangle((0, ground_y_pos - 0.01), 1, 0.01, fc='grey')
    ax_partition.add_patch(ground_wall_patch2)
    ax_partition.legend(loc='upper right', fontsize=8)
    
    # Right: force curve
    ax2.plot(exp_displacement, exp_force, '--k', linewidth=1.5, alpha=0.7, label='Experiment')
    line, = ax2.plot([], [], '-b', linewidth=2, label='Simulation')
    point, = ax2.plot([], [], 'or', markersize=8)
    max_disp_plot = 15.
    max_force_plot = max(max(force_history) if force_history else 0, exp_force.max()) * 1.1
    
    def init_animation():
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal', 'box')
        ax1.set_title(f"Von Mises Stress\n({n_grid}x{n_grid} grid)")
        
        ax_partition.set_xlim(0, 1)
        ax_partition.set_ylim(0, 1)
        ax_partition.set_aspect('equal', 'box')
        ax_partition.set_title("Partition & Contact Activation")
        
        ax2.set_xlim(0, max_disp_plot)
        ax2.set_ylim(0, max_force_plot)
        ax2.set_xlabel("Displacement [mm]")
        ax2.set_ylabel("Compressive Force [N]")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        return (scatter, scatter_p0, scatter_p1, scatter_contact,
                top_wall_patch, top_wall_patch2, 
                ground_wall_patch, ground_wall_patch2, line, point)
    
    def update_animation(frame):
        # Left: stress
        scatter.set_offsets(positions_history[frame])
        stresses = stress_history[frame]
        global_vmax = np.percentile(np.concatenate(stress_history), 99.5)
        scatter.set_array(stresses)
        scatter.set_clim(vmin=0, vmax=global_vmax)
        
        # Middle: partition
        pos = positions_history[frame]
        partition = partition_history[frame]
        
        mask_p0 = (partition == 0)
        mask_p1 = (partition == 1)
        
        scatter_p0.set_offsets(pos[mask_p0])
        scatter_p1.set_offsets(pos[mask_p1])
        scatter_contact.set_offsets(contact_nodes_history[frame])
        
        n_contact = len(contact_nodes_history[frame])
        n_p0 = np.sum(mask_p0)
        n_p1 = np.sum(mask_p1)
        ax_partition.set_title(
            f"Partition & Contact Activation\n"
            f"P0: {n_p0} | P1: {n_p1} | Contact: {n_contact} nodes"
        )
        
        # Compression plate position
        current_comp_step = (frame + 1) * save_every
        current_time = current_comp_step * dt
        plate_y = initial_y_top - compression_velocity * current_time
        top_wall_patch.set_y(plate_y)
        top_wall_patch2.set_y(plate_y)
        
        # Right: force curve
        disps_for_line = [0.0] + displacement_history[:frame+1]
        forces_for_line = [0.0] + force_history[:frame+1]
        line.set_data(disps_for_line, forces_for_line)
        point.set_data([displacement_history[frame]], [force_history[frame]])
        
        return (scatter, scatter_p0, scatter_p1, scatter_contact,
            top_wall_patch, top_wall_patch2,
            ground_wall_patch, ground_wall_patch2, line, point)
    
    ani = FuncAnimation(fig, update_animation, frames=len(positions_history),
                        init_func=init_animation, blit=False, interval=50)
    
    output_filename = 'compression_viscoelastic_grad_v3.gif'
    # ani.save(output_filename, writer='pillow', fps=15)
    print(f"Animation saved as {output_filename}")
    plt.show()


def test_gradient():
    """Test gradient computation and visualize the gradient field."""
    print("\n" + "="*60)
    print("Testing gradient computation...")
    print("="*60)
    
    # Simple test: compute gradient of force with respect to density field
    voxel_2d = np.round(np.load(SHAPE_FILENAME)).astype(float)
    shape_matrix = jnp.array(voxel_2d, dtype=jnp.float64)
    
    n_grid = max(shape_matrix.shape) + 2 * grid_padding_cells
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    
    # 计算初始几何参数
    temp_state = init_particles_from_density(shape_matrix, dx)
    p_x_init = temp_state[0]
    initial_y_top = jnp.max(p_x_init[:, 1])
    ground_y_pos = jnp.min(p_x_init[:, 1])
    
    # Use a small number of steps for testing
    n_steps_test = 50
    
    def objective(density):
        force, _ = run_simulation(density, n_steps_test, n_grid, inv_dx, initial_y_top, ground_y_pos)
        return force
    
    print(f"\nComputing forward pass...")
    start = time.time()
    force_val = objective(shape_matrix)
    end = time.time()
    print(f"Force: {force_val:.4f} N")
    print(f"Forward pass time: {end - start:.2f} seconds")
    
    print(f"\nComputing gradient...")
    start = time.time()
    grad_fn = jax.grad(objective)
    grad_val = grad_fn(shape_matrix)
    end = time.time()
    print(f"Gradient computed!")
    print(f"Gradient shape: {grad_val.shape}")
    print(f"Gradient range: [{jnp.min(grad_val):.6f}, {jnp.max(grad_val):.6f}]")
    print(f"Gradient time: {end - start:.2f} seconds")
    
    print("\nGradient test passed! ✓")
    
    # Finite difference verification (randomly select a few points)
    print("\n" + "="*60)
    print("Finite Difference Verification:")
    print("="*60)
    
    solid_mask = shape_matrix > 0.5
    solid_indices = np.argwhere(solid_mask)
    n_test_points = min(4, len(solid_indices))
    test_indices = solid_indices[np.random.choice(len(solid_indices), n_test_points, replace=False)]
    
    epsilon = 1e-4
    print(f"Testing {n_test_points} random points with ε = {epsilon}")
    print(f"{'Point':>10s} {'AD Grad':>15s} {'FD Grad':>15s} {'Rel Error':>15s}")
    print("-" * 60)
    
    for idx in test_indices:
        i, j = idx
        
        # Automatic differentiation gradient
        ad_grad = float(grad_val[i, j])
        
        # 有限差分梯度
        perturbed_plus = shape_matrix.at[i, j].add(epsilon)
        perturbed_minus = shape_matrix.at[i, j].add(-epsilon)
        
        force_plus = objective(perturbed_plus)
        force_minus = objective(perturbed_minus)
        fd_grad = float((force_plus - force_minus) / (2 * epsilon))
        
        # Relative error
        rel_error = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-10)
        
        print(f"({i:2d},{j:2d})    {ad_grad:15.6e} {fd_grad:15.6e} {100 * rel_error:15.6e}%")
    
    # Visualize gradient field
    print("\n" + "="*60)
    print("Visualizing gradient field...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Original shape
    ax1 = axes[0]
    im1 = ax1.imshow(np.array(shape_matrix), cmap='gray')
    ax1.set_title(f'Original Shape\n(Force = {force_val:.2f} N)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    # 2. Raw gradient field
    ax2 = axes[1]
    grad_np = np.array(grad_val)
    im2 = ax2.imshow(grad_np, cmap='RdBu_r')
    ax2.set_title(f'Gradient Field (dF/dρ)\nRange: [{jnp.min(grad_val):.2e}, {jnp.max(grad_val):.2e}]', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, label='dForce/dDensity')
    
    # 3. Gradient field normalized to [-1, 1]
    ax3 = axes[2]
    grad_max_abs = np.max(np.abs(grad_np))
    grad_normalized = grad_np / (grad_max_abs + 1e-10)
    im3 = ax3.imshow(grad_normalized, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    ax3.set_title(f'Normalized Gradient [-1, 1]\nmax|dF/dρ| = {grad_max_abs:.2e}', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, label='Normalized Gradient')
    
    plt.tight_layout()
    # plt.savefig('gradient_field.png', dpi=150, bbox_inches='tight')
    print("Gradient field saved as 'gradient_field.png'")
    plt.show()
    
    # Statistics
    print("\n" + "="*60)
    print("Gradient Statistics:")
    print("="*60)
    grad_solid = grad_np[solid_mask]
    print(f"Solid cells: {np.sum(solid_mask)}")
    print(f"Gradient mean (solid): {np.mean(grad_solid):.6e}")
    print(f"Gradient std (solid): {np.std(grad_solid):.6e}")
    print(f"Gradient max (solid): {np.max(grad_solid):.6e}")
    print(f"Gradient min (solid): {np.min(grad_solid):.6e}")
    print(f"Non-zero gradients: {np.sum(np.abs(grad_solid) > 1e-10)}")


def test_gradient_curve():
    """Test gradient computation for run_simulation_with_curve."""
    print("\n" + "="*60)
    print("Testing gradient computation for force curve...")
    print("="*60)
    
    # Load voxel shape
    voxel_2d = np.round(np.load(SHAPE_FILENAME)).astype(float)
    shape_matrix = jnp.array(voxel_2d, dtype=jnp.float64)
    
    n_grid = max(shape_matrix.shape) + 2 * grid_padding_cells
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    
    # 计算初始几何参数
    temp_state = init_particles_from_density(shape_matrix, dx)
    p_x_init = temp_state[0]
    initial_y_top = jnp.max(p_x_init[:, 1])
    ground_y_pos = jnp.min(p_x_init[:, 1])
    
    # Simulation parameters
    checkpoint_every = 100
    n_steps_test = 100  # 100 checkpoint points
    
    print(f"\nSimulation parameters:")
    print(f"  Total steps: {n_steps_test}")
    print(f"  Checkpoint every: {checkpoint_every}")
    print(f"  Number of curve points: {n_steps_test // checkpoint_every}")
    
    def objective_curve(density):
        """Objective: sum of all forces along the curve."""
        force_curve, _ = run_simulation_with_curve(
            density, n_steps_test, n_grid, inv_dx, 
            initial_y_top, ground_y_pos, checkpoint_every
        )
        return jnp.sum(force_curve)
    
    # Forward pass
    print(f"\nComputing forward pass...")
    start = time.time()
    force_sum_val = objective_curve(shape_matrix)
    end = time.time()
    print(f"Sum of forces: {force_sum_val:.4f} N")
    print(f"Forward pass time: {end - start:.2f} seconds")
    
    # Compute gradient
    print(f"\nComputing gradient...")
    start = time.time()
    grad_fn = jax.grad(objective_curve)
    grad_val = grad_fn(shape_matrix)
    end = time.time()
    print(f"Gradient computed!")
    print(f"Gradient shape: {grad_val.shape}")
    print(f"Gradient range: [{jnp.min(grad_val):.6f}, {jnp.max(grad_val):.6f}]")
    print(f"Gradient time: {end - start:.2f} seconds")
    
    print("\nGradient test for curve passed! ✓")
    
    # Finite difference verification (randomly select a few points)
    print("\n" + "="*60)
    print("Finite Difference Verification:")
    print("="*60)
    
    solid_mask = shape_matrix > 0.5
    solid_indices = np.argwhere(solid_mask)
    n_test_points = min(8, len(solid_indices))  # Only test a few points because each run is a full simulation
    test_indices = solid_indices[np.random.choice(len(solid_indices), n_test_points, replace=False)]
    
    epsilon = 1e-4
    print(f"Testing {n_test_points} random points with ε = {epsilon}")
    print(f"{'Point':>10s} {'AD Grad':>15s} {'FD Grad':>15s} {'Rel Error':>15s}")
    print("-" * 60)
    
    for idx in test_indices:
        i, j = idx
        
        # Automatic differentiation gradient
        ad_grad = float(grad_val[i, j])
        
        # 有限差分梯度
        perturbed_plus = shape_matrix.at[i, j].add(epsilon)
        perturbed_minus = shape_matrix.at[i, j].add(-epsilon)
        
        force_plus = objective_curve(perturbed_plus)
        force_minus = objective_curve(perturbed_minus)
        fd_grad = float((force_plus - force_minus) / (2 * epsilon))
        
        # Relative error
        rel_error = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-10)
        
        print(f"({i:2d},{j:2d})    {ad_grad:15.6e} {fd_grad:15.6e} {100 * rel_error:15.6e}%")
    
    # Visualize gradient field
    print("\n" + "="*60)
    print("Visualizing gradient field...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Original shape
    ax1 = axes[0]
    im1 = ax1.imshow(np.array(shape_matrix), cmap='gray')
    ax1.set_title(f'Original Shape\n(Sum of Forces = {force_sum_val:.2f} N)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    # 2. Raw gradient field
    ax2 = axes[1]
    grad_np = np.array(grad_val)
    im2 = ax2.imshow(grad_np, cmap='RdBu_r')
    ax2.set_title(f'Gradient Field (d∑F/dρ)\nRange: [{jnp.min(grad_val):.2e}, {jnp.max(grad_val):.2e}]', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, label='d(SumForce)/dDensity')
    
    # 3. Gradient field normalized to [-1, 1]
    ax3 = axes[2]
    grad_max_abs = np.max(np.abs(grad_np))
    grad_normalized = grad_np / (grad_max_abs + 1e-10)
    im3 = ax3.imshow(grad_normalized, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    ax3.set_title(f'Normalized Gradient [-1, 1]\nmax|d∑F/dρ| = {grad_max_abs:.2e}', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, label='Normalized Gradient')
    
    plt.tight_layout()
    plt.savefig('gradient_field_curve.png', dpi=150, bbox_inches='tight')
    print("Gradient field saved as 'gradient_field_curve.png'")
    plt.show()
    
    # Statistics
    print("\n" + "="*60)
    print("Gradient Statistics:")
    print("="*60)
    grad_solid = grad_np[solid_mask]
    print(f"Solid cells: {np.sum(solid_mask)}")
    print(f"Gradient mean (solid): {np.mean(grad_solid):.6e}")
    print(f"Gradient std (solid): {np.std(grad_solid):.6e}")
    print(f"Gradient max (solid): {np.max(grad_solid):.6e}")
    print(f"Gradient min (solid): {np.min(grad_solid):.6e}")
    print(f"Non-zero gradients: {np.sum(np.abs(grad_solid) > 1e-10)}")


if __name__ == '__main__':
    main()
    # test_gradient()
    # test_gradient_curve()

