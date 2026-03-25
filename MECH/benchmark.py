"""
Corrected implementation based on:
- Homel & Herbold 2017: Field-gradient partitioning for fracture and frictional contact
- Xiao, Liu & Sun 2021: DP-MPM for evolving multi-body thermal-mechanical contacts

Key corrections:
1. Proper mass gradient computation for contact normals
2. Correct separability condition for self-contact
3. Fixed contact force calculation following both papers
4. Proper field partitioning based on damage gradient
"""

import os
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
jax.config.update("jax_enable_x64", True)

# ==================== Parameters ====================
dim = 2

E = 1.0e8
nu = 0.3
rho = 2.3e3

mu = E / (2.0 * (1.0 + nu))
lambda_param = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

dt = 1e-4
damping = 0.01

n_grid_x = 45
n_grid_y = 30
dx = 1.0 / n_grid_y
inv_dx = float(n_grid_y)

pentagon_width = 0.5
pentagon_height = 1.0
pentagon_shoulder_ratio = 0.2
pentagon_bottom_width = 0.1

block_size = 0.5

disk_center_x = 0.7
disk_center_y = 0.5
disk_radius = 0.27

gap = 0.00
pentagon_x_min = 0.0
domain_width = n_grid_x * dx
block_x_max = domain_width
grid_limit_y = n_grid_y * dx

applied_stress = 3.e6
force_duration_steps = 30000
friction_coef = 0.5

ground_y = 0.0
top_bound_y = 1.0
particles_per_cell = 3
n_steps = 1500
save_every = 10
gravity = 0.0

# Separability thresholds (Homel 2017 Section 2.3)
D_min = 0.25  # Minimum average damage for separability
D_cr = 0.5    # Critical damage threshold


def safe_norm(x, axis=-1, keepdims=False, eps=1e-15):
    """Numerically stable norm computation"""
    sq_sum = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(sq_sum + eps)


def quadratic_bspline_weights(fx):
    """Quadratic B-spline weights for MPM"""
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return w0, w1, w2


def quadratic_bspline_grad_weights(fx):
    """Gradient of quadratic B-spline weights"""
    dw0 = -(1.5 - fx)
    dw1 = -2.0 * (fx - 1.0)
    dw2 = (fx - 0.5)
    return dw0, dw1, dw2


def cubic_kernel_paper_eq18(r_normalized):
    """Cubic kernel from Homel 2017 Eq. 18"""
    r = jnp.abs(r_normalized)
    val = 1.0 - 3.0 * r**2 + 2.0 * r**3
    return jnp.where(r <= 1.0, val, 0.0)


def cubic_kernel_derivative_eq18(r_normalized):
    """Derivative of cubic kernel"""
    r = jnp.abs(r_normalized)
    val = -6.0 * r + 6.0 * r**2
    return jnp.where(r <= 1.0, val, 0.0)


# ==================== Initialization Functions ====================

def create_pentagon_particles(x_min, y_min, width, height, shoulder_ratio, bottom_width):
    sub_spacing = dx / particles_per_cell
    n_x = int(width / sub_spacing)
    n_y = int(height / sub_spacing)
    
    positions = []
    slanted_height = height * (1.0 - shoulder_ratio)
    
    for i in range(n_x):
        for j in range(n_y):
            x_local = i * sub_spacing
            y_local = j * sub_spacing
            
            if y_local >= slanted_height:
                width_at_y = width
            else:
                ratio = y_local / slanted_height
                width_at_y = bottom_width + (width - bottom_width) * ratio
            
            if x_local <= width_at_y:
                x = x_min + x_local
                y = y_min + y_local
                positions.append([x, y])
                
    return jnp.array(positions)


def create_rect_particles(x_min, y_min, width, height):
    sub_spacing = dx / particles_per_cell
    n_x = int(width / sub_spacing)
    n_y = int(height / sub_spacing)
    
    positions = []
    for i in range(n_x):
        for j in range(n_y):
            x = x_min + i * sub_spacing
            y = y_min + j * sub_spacing
            positions.append([x, y])
    return jnp.array(positions)


def create_disk_particles(center_x, center_y, radius):
    sub_spacing = dx / particles_per_cell
    n_samples = int(2 * radius / sub_spacing)
    
    positions = []
    for i in range(n_samples):
        for j in range(n_samples):
            x_local = (i + 0.5) * sub_spacing - radius
            y_local = (j + 0.5) * sub_spacing - radius
            dist = jnp.sqrt(x_local**2 + y_local**2)
            if dist < radius:
                x = center_x + x_local
                y = center_y + y_local
                positions.append([x, y])
    
    return jnp.array(positions)


def detect_surface_particles(positions, r_kernel, inv_r_kernel):
    """
    Surface detection using kernel sum (Homel 2017 Section 2.4, Figure 4b)
    S(x) = Σ_p ω(|x - x_p|/r_p)
    Surface particles have lower S values
    """
    n_particles = positions.shape[0]
    
    # Compute kernel sum at each particle location
    kernel_sum = jnp.zeros(n_particles)
    
    for p in range(n_particles):
        dist_vec = positions - positions[p]
        dist = safe_norm(dist_vec, axis=-1)
        r_bar = dist * inv_r_kernel
        weight = cubic_kernel_paper_eq18(r_bar)
        kernel_sum = kernel_sum.at[p].set(jnp.sum(weight))
    
    # Surface particles have kernel sum below threshold
    kernel_sum_max = jnp.max(kernel_sum)
    kernel_sum_threshold = 0.8 * kernel_sum_max
    surface_flag = jnp.where(kernel_sum < kernel_sum_threshold, 1.0, 0.0)
    
    n_surface = jnp.sum(surface_flag > 0.5)
    print(f"  Surface particles: {int(n_surface)} / {n_particles} ({100.0 * n_surface / n_particles:.1f}%)")
    
    return surface_flag


def initialize_particles():
    pentagon_h_real = top_bound_y - ground_y
    p_pentagon = create_pentagon_particles(
        pentagon_x_min, ground_y, pentagon_width, pentagon_h_real, 
        pentagon_shoulder_ratio, pentagon_bottom_width
    )
    
    block_x_min = block_x_max - block_size
    p_block = create_rect_particles(block_x_min, ground_y, block_size, block_size)
    
    p_disk = create_disk_particles(disk_center_x, disk_center_y, disk_radius)

    p_x = jnp.vstack([p_pentagon, p_disk, p_block])
    n_particles = p_x.shape[0]
    
    n_pentagon = p_pentagon.shape[0]
    n_disk = p_disk.shape[0]
    n_block = p_block.shape[0]
    
    body_ids = jnp.concatenate([
        jnp.zeros(n_pentagon),
        jnp.ones(n_disk),
        jnp.ones(n_block) * 2
    ])
    
    print(f"Particles: pentagon={n_pentagon}, disk={n_disk}, block={n_block}, total={n_particles}")
    
    pentagon_mask = jnp.concatenate([
        jnp.ones(n_pentagon, dtype=bool),
        jnp.zeros(n_disk, dtype=bool),
        jnp.zeros(n_block, dtype=bool)
    ])
    
    p_v = jnp.zeros((n_particles, dim))
    p_F = jnp.tile(jnp.eye(dim)[None, :, :], (n_particles, 1, 1))
    p_C = jnp.zeros((n_particles, dim, dim))
    
    p_vol_0 = (dx / particles_per_cell) ** 2
    p_mass = p_vol_0 * rho
    p_vol0 = jnp.full(n_particles, p_vol_0)
    p_mass_arr = jnp.full(n_particles, p_mass)
    
    # Surface detection
    r_kernel = jnp.sqrt(2.0) * dx
    inv_r_kernel = 1.0 / r_kernel
    p_surface_flag = detect_surface_particles(p_x, r_kernel, inv_r_kernel)
    
    # Identify left surface particles for force application
    pentagon_surface = p_surface_flag[:n_pentagon] > 0.5
    pentagon_positions = p_x[:n_pentagon]
    
    n_y_bins = 20
    y_min = jnp.min(pentagon_positions[:, 1])
    y_max = jnp.max(pentagon_positions[:, 1])
    y_bin_width = (y_max - y_min) / n_y_bins
    
    left_surface_mask = jnp.zeros(n_particles, dtype=bool)
    
    for iy in range(n_y_bins):
        y_low = y_min + iy * y_bin_width
        y_high = y_low + y_bin_width
        
        in_y_bin = (pentagon_positions[:, 1] >= y_low) & (pentagon_positions[:, 1] < y_high) & pentagon_surface
        
        if jnp.any(in_y_bin):
            x_in_bin = jnp.where(in_y_bin, pentagon_positions[:, 0], jnp.inf)
            x_min_in_bin = jnp.min(x_in_bin)
            threshold = x_min_in_bin + 0.1 * dx
            is_left_in_bin = in_y_bin & (pentagon_positions[:, 0] <= threshold)
            left_surface_mask = left_surface_mask.at[:n_pentagon].set(
                left_surface_mask[:n_pentagon] | is_left_in_bin
            )
    
    n_left_surface = jnp.sum(left_surface_mask)
    print(f"Pentagon left surface particles: {int(n_left_surface)}")
    
    total_area = (y_max - y_min) * 1.0
    total_force = applied_stress * total_area
    force_per_particle = total_force / (n_left_surface + 1e-10)
    
    return (p_x, p_v, p_F, p_C, p_mass_arr, p_vol0, p_surface_flag, 
            body_ids, pentagon_mask, left_surface_mask, force_per_particle)


# ==================== MPM Step ====================

@jit
def mpm_step(p_x, p_v, p_F, p_C, p_mass, p_vol0, p_surface_flag, body_ids, 
             pentagon_mask, left_surface_mask, force_per_particle, 
             force_duration_steps, step):
    """
    MPM step with DFG partitioning for self-contact
    
    Key references:
    - Homel 2017: Equations 11, 12, 14-16, 17-21, Algorithm 1-3
    - Xiao 2021: Equations 72-77
    """
    
    n_particles = p_x.shape[0]
    
    # Kernel parameters
    r_kernel = jnp.sqrt(2.0) * dx
    inv_r_kernel = 1.0 / r_kernel
    
    # For self-contact: damage = surface flag (Homel 2017 Section 2.4)
    # D(x) uses max(s_p, D_p) where s_p is surface flag
    p_damage = p_surface_flag
    
    base = (p_x * inv_dx - 0.5).astype(jnp.int32)
    limit_array = jnp.array([n_grid_x - 1, n_grid_y - 1])
    
    # ========================================
    # Step 1: Construct normalized damage field D̄(x) (Eq. 17, 19)
    # D̄(x) = D(x) / S(x)
    # ========================================
    grid_damage_sum = jnp.zeros((n_grid_x, n_grid_y))
    grid_weight_sum = jnp.zeros((n_grid_x, n_grid_y))
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            node_pos = grid_idx.astype(jnp.float64) * dx
            dist_vec = p_x - node_pos
            dist = safe_norm(dist_vec, axis=-1)
            r_bar = dist * inv_r_kernel
            weight = cubic_kernel_paper_eq18(r_bar)
            
            grid_damage_sum = grid_damage_sum.at[
                grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]
            ].add(weight * p_damage)
            
            grid_weight_sum = grid_weight_sum.at[
                grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]
            ].add(weight)
    
    grid_damage_normalized = jnp.where(
        grid_weight_sum > 1e-15,
        grid_damage_sum / grid_weight_sum,
        0.0
    )
    
    # ========================================
    # Step 2: Compute damage gradient at particles ∇D̄_p
    # ========================================
    p_damage_grad = jnp.zeros((n_particles, 2))
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            node_pos = grid_idx.astype(jnp.float64) * dx
            d_node = grid_damage_normalized[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            
            dist_vec = p_x - node_pos
            dist = safe_norm(dist_vec, axis=-1)
            r_bar = dist * inv_r_kernel
            
            # Gradient of kernel
            dw_dr = cubic_kernel_derivative_eq18(r_bar)
            grad_w_coef = jnp.where(dist > 1e-15, dw_dr * inv_r_kernel / dist, 0.0)
            grad_w = dist_vec * grad_w_coef[:, None]
            
            p_damage_grad = p_damage_grad + d_node[:, None] * grad_w
    
    # ========================================
    # Step 3: Compute nonlocal grid damage gradient ∇D̄*_i (Algorithm 1)
    # Select particle with maximum gradient magnitude
    # ========================================
    fx = p_x * inv_dx - base.astype(jnp.float64)
    w0, w1, w2 = quadratic_bspline_weights(fx)
    weights = jnp.stack([w0, w1, w2], axis=1)
    
    dw0, dw1, dw2 = quadratic_bspline_grad_weights(fx)
    dweights = jnp.stack([dw0, dw1, dw2], axis=1)
    
    p_grad_norm = safe_norm(p_damage_grad, axis=-1)
    
    grid_max_grad_norm = jnp.full((n_grid_x, n_grid_y), -1e10)
    grid_damage_grad_nonlocal = jnp.zeros((n_grid_x, n_grid_y, 2))
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            current_max_norm = grid_max_grad_norm[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            should_update = p_grad_norm > current_max_norm
            
            new_max_norm = jnp.where(should_update, p_grad_norm, current_max_norm)
            current_grad = grid_damage_grad_nonlocal[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            new_grad = jnp.where(should_update[:, None], p_damage_grad, current_grad)
            
            grid_max_grad_norm = grid_max_grad_norm.at[
                grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]
            ].set(new_max_norm)
            
            grid_damage_grad_nonlocal = grid_damage_grad_nonlocal.at[
                grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]
            ].set(new_grad)
    
    # ========================================
    # Step 4: P2G with DFG partitioning (Algorithm 2)
    # Partition based on sgn(∇D̄_p · ∇D̄*_i)
    # ========================================
    grid_m = jnp.zeros((2, n_grid_x, n_grid_y))
    grid_p = jnp.zeros((2, n_grid_x, n_grid_y, 2))
    grid_mass_grad = jnp.zeros((2, n_grid_x, n_grid_y, 2))  # For contact normals
    grid_avg_damage = jnp.zeros((2, n_grid_x, n_grid_y))    # For separability
    grid_surface_mass = jnp.zeros((2, n_grid_x, n_grid_y))  # Surface mass per field
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            weight = weights[:, i, 0] * weights[:, j, 1]
            
            # Shape function gradients
            grad_weight_x = dweights[:, i, 0] * weights[:, j, 1] * inv_dx
            grad_weight_y = weights[:, i, 0] * dweights[:, j, 1] * inv_dx
            grad_weight = jnp.stack([grad_weight_x, grad_weight_y], axis=-1)
            
            dpos = (offset.astype(jnp.float64) - fx) * dx
            
            # Field partitioning based on gradient dot product
            grad_at_node = grid_damage_grad_nonlocal[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            dot_prod = jnp.einsum('pi,pi->p', p_damage_grad, grad_at_node)
            field_idx = jnp.where(dot_prod >= 0.0, 0, 1)
            
            mask_0 = (field_idx == 0)
            mask_1 = (field_idx == 1)
            
            # Velocity at grid position (with APIC)
            v_at_grid = p_v + jnp.einsum('pij,pj->pi', p_C, dpos)
            momentum = p_mass[:, None] * v_at_grid
            mass_contrib = weight * p_mass
            mom_contrib = weight[:, None] * momentum
            
            # Mass gradient contribution (Homel 2017 Eq. 11)
            mass_grad_contrib = p_mass[:, None] * grad_weight
            
            # Damage-weighted mass for average damage computation
            damage_mass_contrib = weight * p_mass * p_damage
            
            # Surface mass contribution
            surface_mass_contrib = weight * p_mass * p_surface_flag
            
            # Accumulate to field 0
            grid_m = grid_m.at[0, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_0, mass_contrib, 0.0)
            )
            grid_p = grid_p.at[0, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_0[:, None], mom_contrib, 0.0)
            )
            grid_mass_grad = grid_mass_grad.at[0, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_0[:, None], mass_grad_contrib, 0.0)
            )
            grid_avg_damage = grid_avg_damage.at[0, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_0, damage_mass_contrib, 0.0)
            )
            grid_surface_mass = grid_surface_mass.at[0, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_0, surface_mass_contrib, 0.0)
            )
            
            # Accumulate to field 1
            grid_m = grid_m.at[1, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_1, mass_contrib, 0.0)
            )
            grid_p = grid_p.at[1, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_1[:, None], mom_contrib, 0.0)
            )
            grid_mass_grad = grid_mass_grad.at[1, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_1[:, None], mass_grad_contrib, 0.0)
            )
            grid_avg_damage = grid_avg_damage.at[1, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_1, damage_mass_contrib, 0.0)
            )
            grid_surface_mass = grid_surface_mass.at[1, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(
                jnp.where(mask_1, surface_mass_contrib, 0.0)
            )
    
    # Compute average damage per field (Homel 2017 Eq. 21)
    grid_avg_damage = jnp.where(
        grid_m > 1e-15,
        grid_avg_damage / (grid_m + 1e-15),
        0.0
    )
    
    # ========================================
    # Step 5: Compute initial velocities v̄_ζi
    # ========================================
    v_bar = jnp.zeros((2, n_grid_x, n_grid_y, 2))
    for zeta in range(2):
        v_bar = v_bar.at[zeta].set(
            jnp.where(grid_m[zeta, :, :, None] > 1e-15,
                     grid_p[zeta] / (grid_m[zeta, :, :, None] + 1e-15),
                     0.0)
        )
    
    # ========================================
    # Step 6: Compute internal forces and update velocities
    # ========================================
    J = p_F[:, 0, 0] * p_F[:, 1, 1] - p_F[:, 0, 1] * p_F[:, 1, 0]
    J_safe = jnp.maximum(J, 1e-9)
    
    F_inv_T = jnp.stack([
        jnp.stack([p_F[:, 1, 1], -p_F[:, 1, 0]], axis=-1),
        jnp.stack([-p_F[:, 0, 1], p_F[:, 0, 0]], axis=-1)
    ], axis=1) / J_safe[:, None, None]
    
    # Neo-Hookean stress
    P = mu * (p_F - F_inv_T) + lambda_param * jnp.log(J_safe)[:, None, None] * F_inv_T
    kirchhoff_stress = jnp.einsum('pij,pkj->pik', P, p_F)
    
    D_inv = 4.0 / (dx * dx)
    stress_term = -D_inv * p_vol0[:, None, None] * kirchhoff_stress
    
    grid_f_total = jnp.zeros((n_grid_x, n_grid_y, dim))
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            weight = weights[:, i, 0] * weights[:, j, 1]
            dpos = (offset.astype(jnp.float64) - fx) * dx
            
            force_contrib = weight[:, None] * jnp.einsum('pij,pj->pi', stress_term, dpos)
            grid_f_total = grid_f_total.at[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(force_contrib)
    
    # Apply external force
    force_active = (step < force_duration_steps)
    applied_force_vector = jnp.where(
        force_active,
        jnp.array([force_per_particle, 0.0]),
        jnp.array([0.0, 0.0])
    )
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            weight = weights[:, i, 0] * weights[:, j, 1]
            traction_contrib = jnp.where(
                left_surface_mask[:, None],
                weight[:, None] * applied_force_vector[None, :],
                0.0
            )
            grid_f_total = grid_f_total.at[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]].add(traction_contrib)
    
    # Distribute forces to fields by mass ratio
    m_total = grid_m[0] + grid_m[1] + 1e-15
    grid_f = jnp.zeros((2, n_grid_x, n_grid_y, 2))
    for zeta in range(2):
        grid_f = grid_f.at[zeta].set(
            grid_f_total * (grid_m[zeta] / m_total)[:, :, None]
        )
    
    # Update velocities (pre-contact)
    v_hat = jnp.zeros((2, n_grid_x, n_grid_y, 2))
    for zeta in range(2):
        a_zeta = grid_f[zeta] / (grid_m[zeta, :, :, None] + 1e-15) + jnp.array([0.0, gravity])
        v_hat = v_hat.at[zeta].set(v_bar[zeta] + dt * a_zeta)
    
    v_hat = v_hat * (1.0 - damping)
    
    # ========================================
    # Step 7: Contact detection and force calculation
    # Based on Xiao 2021 Eq. 72-77 and Homel 2017 Eq. 11-16
    # ========================================
    
    # Check if both fields have mass at this node
    both_fields_present = (grid_m[0] > 1e-10) & (grid_m[1] > 1e-10)
    
    # Separability condition (Homel 2017 Eq. 71, Section 2.3)
    # For self-contact: check if surface particles are present
    has_surface_0 = grid_surface_mass[0] > 1e-15
    has_surface_1 = grid_surface_mass[1] > 1e-15
    
    # Alternative: use average damage for separability
    # D_avg_0 = grid_avg_damage[0]
    # D_avg_1 = grid_avg_damage[1]
    # max_avg_damage = jnp.maximum(D_avg_0, D_avg_1)
    # min_avg_damage = jnp.minimum(D_avg_0, D_avg_1)
    # is_separable = both_fields_present & (max_avg_damage > D_cr) & (min_avg_damage > D_min)
    
    # For self-contact, separability is based on surface particles
    is_separable = both_fields_present & (has_surface_0 | has_surface_1)
    
    # Compute contact normals (Homel 2017 Eq. 11, Xiao 2021 Eq. 74)
    # n̂_k = Σ_kp m_p ∇S_ip / |Σ_kp m_p ∇S_ip|
    n_hat_0 = grid_mass_grad[0] / (safe_norm(grid_mass_grad[0], axis=-1, keepdims=True) + 1e-15)
    n_hat_1 = grid_mass_grad[1] / (safe_norm(grid_mass_grad[1], axis=-1, keepdims=True) + 1e-15)
    
    # Corrected surface normals (Homel 2017: n*_1 = n*_2 = (n̂_1 - n̂_2)/2, then normalize)
    n_diff = n_hat_0 - n_hat_1
    n_diff_norm = safe_norm(n_diff, axis=-1, keepdims=True)
    n_1 = n_diff / (n_diff_norm + 1e-15)  # Normal for field 0
    n_2 = -n_1                              # Normal for field 1
    
    # Center-of-mass velocity (Xiao 2021 Eq. 72)
    m0, m1 = grid_m[0], grid_m[1]
    m_sum = m0 + m1 + 1e-15
    v_cm = (m0[:, :, None] * v_hat[0] + m1[:, :, None] * v_hat[1]) / m_sum[:, :, None]
    
    # Check for approaching contact (Xiao 2021 Eq. 73)
    # Contact occurs if (v̂_ki - v^cm) · n_ki > 0
    approach_0 = jnp.sum((v_hat[0] - v_cm) * n_1, axis=-1)
    approach_1 = jnp.sum((v_hat[1] - v_cm) * n_2, axis=-1)
    




    #################################################### super important here
    is_approaching = (approach_0 > 0) | (approach_1 > 0) 
    
    valid_normal = n_diff_norm[:, :, 0] > 1e-6
    apply_contact = is_separable & is_approaching & valid_normal
    
    # Contact velocity correction
    v_star = v_hat.copy()
    
    # Field 0 correction (Xiao 2021 Eq. 75-77)
    delta_v_0 = v_cm - v_hat[0]
    v_n_0 = jnp.sum(delta_v_0 * n_1, axis=-1)  # Normal component
    delta_v_t_0 = delta_v_0 - v_n_0[:, :, None] * n_1  # Tangential component
    v_t_0_mag = safe_norm(delta_v_t_0, axis=-1)
    t_dir_0 = delta_v_t_0 / (v_t_0_mag[:, :, None] + 1e-15)
    
    # Coulomb friction (Xiao 2021 Eq. 76)
    max_friction_0 = friction_coef * jnp.abs(v_n_0)
    v_t_0_corrected = jnp.minimum(v_t_0_mag, max_friction_0)
    
    # Apply correction only when approaching
    correction_0 = v_n_0[:, :, None] * n_1 + v_t_0_corrected[:, :, None] * t_dir_0
    should_correct_0 = apply_contact #& (v_n_0 > 0) #################################################### super important here
    v_star = v_star.at[0].set(
        jnp.where(should_correct_0[:, :, None], v_hat[0] + correction_0, v_hat[0])
    )
    
    # Field 1 correction (symmetric)
    delta_v_1 = v_cm - v_hat[1]
    v_n_1 = jnp.sum(delta_v_1 * n_2, axis=-1)
    delta_v_t_1 = delta_v_1 - v_n_1[:, :, None] * n_2
    v_t_1_mag = safe_norm(delta_v_t_1, axis=-1)
    t_dir_1 = delta_v_t_1 / (v_t_1_mag[:, :, None] + 1e-15)
    
    max_friction_1 = friction_coef * jnp.abs(v_n_1)
    v_t_1_corrected = jnp.minimum(v_t_1_mag, max_friction_1)
    
    correction_1 = v_n_1[:, :, None] * n_2 + v_t_1_corrected[:, :, None] * t_dir_1
    should_correct_1 = apply_contact #& (v_n_1 > 0)  #################################################### super important here
    v_star = v_star.at[1].set(
        jnp.where(should_correct_1[:, :, None], v_hat[1] + correction_1, v_hat[1])
    )
    
    # ========================================
    # Step 8: Boundary conditions
    # ========================================
    for zeta in range(2):
        # Bottom and top walls
        v_star = v_star.at[zeta, :, 0, 1].set(0.0)
        v_star = v_star.at[zeta, :, n_grid_y-1, 1].set(0.0)
        
        # Left wall (allow outward motion)
        v_star = v_star.at[zeta, 0, :, 0].set(jnp.maximum(v_star[zeta, 0, :, 0], 0.0))
        
        # Right wall (allow inward motion)
        v_star = v_star.at[zeta, n_grid_x-1, :, 0].set(jnp.minimum(v_star[zeta, n_grid_x-1, :, 0], 0.0))
    
    # ========================================
    # Step 9: G2P transfer
    # ========================================
    new_p_v = jnp.zeros_like(p_v)
    p_B = jnp.zeros_like(p_C)
    
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            weight = weights[:, i, 0] * weights[:, j, 1]
            
            # Get correct velocity field for each particle
            grad_at_node = grid_damage_grad_nonlocal[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            dot_prod = jnp.einsum('pi,pi->p', p_damage_grad, grad_at_node)
            field_idx = jnp.where(dot_prod >= 0.0, 0, 1)
            
            v_node_0 = v_star[0, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            v_node_1 = v_star[1, grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            v_on_grid = jnp.where(field_idx[:, None] == 0, v_node_0, v_node_1)
            
            dpos = (offset.astype(jnp.float64) - fx) * dx
            new_p_v = new_p_v + weight[:, None] * v_on_grid
            p_B = p_B + weight[:, None, None] * jnp.einsum('pi,pj->pij', v_on_grid, dpos)
    
    # Update APIC affine matrix
    new_p_C = p_B * 4.0 * inv_dx * inv_dx
    
    # Update deformation gradient
    F_update = jnp.eye(dim) + dt * new_p_C
    new_p_F = jnp.einsum('pij,pjk->pik', F_update, p_F)
    
    # Update positions
    new_p_x = p_x + dt * new_p_v
    
    # Position boundary constraints
    margin = 0.01
    new_p_x = jnp.stack([
        jnp.clip(new_p_x[:, 0], margin, domain_width - margin),
        jnp.clip(new_p_x[:, 1], margin, grid_limit_y - margin)
    ], axis=-1)
    
    # Compute field assignment for visualization
    partition_scores = jnp.zeros(n_particles)
    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            grid_idx = base + offset
            grid_idx_clipped = jnp.clip(grid_idx, 0, limit_array)
            
            weight = weights[:, i, 0] * weights[:, j, 1]
            grad_at_node = grid_damage_grad_nonlocal[grid_idx_clipped[:, 0], grid_idx_clipped[:, 1]]
            dot_prod = jnp.einsum('pi,pi->p', p_damage_grad, grad_at_node)
            partition_scores = partition_scores + weight * dot_prod
    
    p_field_id = jnp.where(partition_scores >= 0, 0, 1).astype(jnp.int32)
    
    return new_p_x, new_p_v, new_p_F, new_p_C, p_field_id


# ==================== Main Functions ====================

def run_simulation():
    (p_x, p_v, p_F, p_C, p_mass, p_vol0, p_surface_flag, 
     body_ids, pentagon_mask, left_surface_mask, force_per_particle) = initialize_particles()
    
    history = {
        'p_x': [],
        'p_v': [],
        'p_field_id': [],
        'p_surface_flag': p_surface_flag,
        'body_ids': body_ids
    }
    
    print(f"\nRunning simulation for {n_steps} steps...")
    
    for step in range(n_steps):
        p_x, p_v, p_F, p_C, p_field_id = mpm_step(
            p_x, p_v, p_F, p_C, p_mass, p_vol0, 
            p_surface_flag, body_ids, pentagon_mask, left_surface_mask, 
            force_per_particle, force_duration_steps, step
        )
        
        if step % save_every == 0:
            history['p_x'].append(np.array(p_x))
            history['p_v'].append(np.array(p_v))
            history['p_field_id'].append(np.array(p_field_id))
            
            if step % 1000 == 0:
                v_mag = np.linalg.norm(p_v, axis=1).mean()
                force_status = "ON" if step < force_duration_steps else "OFF"
                print(f"Step {step}/{n_steps}, avg velocity: {v_mag:.6f}, Force: {force_status}")
    
    print("Simulation complete!")
    
    # Save velocity field data for custom plotting
    np.save('velocity_field_positions0.npy', np.array(history['p_x']))
    np.save('velocity_field_velocities0.npy', np.array(history['p_v']))
    print(f"Saved velocity field data:")
    print(f"  - velocity_field_positions.npy: shape {np.array(history['p_x']).shape}")
    print(f"  - velocity_field_velocities.npy: shape {np.array(history['p_v']).shape}")
    
    return history


def create_animation(history):
    fig = plt.figure(figsize=(22, 7), dpi=120)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    body_ids = history['body_ids']
    p_surface_flag = history['p_surface_flag']
    
    scatter_interior = ax0.scatter([], [], s=1, c='lightgray', alpha=0.4, label='Interior')
    scatter_surface = ax0.scatter([], [], s=2, c='red', alpha=0.8, label='Surface')
    ax0.set_xlim(0, domain_width)
    ax0.set_ylim(0, grid_limit_y)
    ax0.set_aspect('equal', 'box')
    ax0.set_title(f'Surface Detection\n({n_grid_x}x{n_grid_y} grid)')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.legend(loc='upper right', fontsize=8)
    ax0.grid(True, alpha=0.3)
    
    scatter_p0 = ax1.scatter([], [], s=2, c='blue', alpha=0.6, label='Field 0')
    scatter_p1 = ax1.scatter([], [], s=2, c='red', alpha=0.6, label='Field 1')
    ax1.set_xlim(0, domain_width)
    ax1.set_ylim(0, grid_limit_y)
    ax1.set_aspect('equal', 'box')
    ax1.set_title('DFG Partitioning')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    scatter_vel = ax2.scatter([], [], s=1, c=[], cmap='viridis')
    ax2.set_xlim(0, domain_width)
    ax2.set_ylim(0, grid_limit_y)
    ax2.set_aspect('equal', 'box')
    ax2.set_title('Velocity Magnitude')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    cbar2 = fig.colorbar(scatter_vel, ax=ax2)
    cbar2.set_label('|v| (m/s)')
    
    def update(frame):
        p_x = history['p_x'][frame]
        p_v = history['p_v'][frame]
        p_field_id = history['p_field_id'][frame]
        
        v_mag = np.linalg.norm(p_v, axis=1)
        is_surface = np.array(p_surface_flag) > 0.5
        
        scatter_interior.set_offsets(p_x[~is_surface])
        scatter_surface.set_offsets(p_x[is_surface])
        
        mask_p0 = (p_field_id == 0)
        mask_p1 = (p_field_id == 1)
        scatter_p0.set_offsets(p_x[mask_p0])
        scatter_p1.set_offsets(p_x[mask_p1])
        
        n_p0 = np.sum(mask_p0)
        n_p1 = np.sum(mask_p1)
        ax1.set_title(f'DFG Partitioning\nField0: {n_p0} | Field1: {n_p1}')
        
        scatter_vel.set_offsets(p_x)
        scatter_vel.set_array(v_mag)
        global_vmax = 0.15
        scatter_vel.set_clim(vmin=0, vmax=global_vmax)
        
        fig.suptitle(f'Frame {frame}/{len(history["p_x"])} - Self-Contact with DFG', 
                    fontsize=14, y=0.98)
        
        return scatter_interior, scatter_surface, scatter_p0, scatter_p1, scatter_vel
    
    n_frames = len(history['p_x'])
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=50, repeat=True)

    output_filename = 'benchmark.gif'
    # ani.save(output_filename, writer='pillow', fps=15)
    
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("DFG Self-Contact Benchmark (Corrected Implementation)")
    print("Based on Homel & Herbold 2017 and Xiao, Liu & Sun 2021")
    print("=" * 60)
    
    history = run_simulation()
    create_animation(history)