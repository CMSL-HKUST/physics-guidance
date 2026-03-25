import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.serialization
import matplotlib.pyplot as plt
import time
from jax import random
from pathlib import Path
from flax.training.train_state import TrainState

from unet import UNet

from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem
from therm.fem_heat import HeatConduction
from jax_fem.solver import ad_wrapper


def main():

    # ----- initialization of unet -----
    IMG_SIZE = 64
    NUM_STEPS = 50
    LEARNING_RATE = 1e-4
    MODEL_PATH = "diffusion/models/vpsde_model.flax"
    
    key = jax.random.PRNGKey(int(time.time()))

    model = UNet()
    dummy_x = jnp.ones((1, IMG_SIZE, IMG_SIZE, 1), dtype=jnp.float64)
    dummy_t = jnp.ones((1,), dtype=jnp.float64)
    key, model_key = random.split(key)
    params = model.init(model_key, dummy_x, dummy_t)['params']
    optimizer = optax.adam(LEARNING_RATE)

    with open(MODEL_PATH, "rb") as f:
        params_bytes = f.read()
    restored_params = flax.serialization.from_bytes(params, params_bytes)

    restored_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if hasattr(x, 'astype') else x, 
        restored_params
    )
    print("model params loaded and converted to float64.")

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=restored_params,
        tx=optimizer
    )


    # ----- initialization of fem heat transfer -----
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.
    Nx, Ny = 64, 64
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    k0 = 1.
    k1 = 100.
    k_target = 30.
    T_top = 500.
    T_bot = 300.

    # --- boundary conditions ---
    def left(point): return jnp.isclose(point[0], 0., atol=1e-5)
    def right(point): return jnp.isclose(point[0], Lx, atol=1e-5)
    def bottom(point): return jnp.isclose(point[1], 0., atol=1e-5)
    def top(point): return jnp.isclose(point[1], Ly, atol=1e-5)

    def dirichlet_val_top(point): return T_top
    def dirichlet_val_bottom(point): return T_bot

    location_fns_dirichlet = [top, bottom]
    value_fns = [dirichlet_val_top, dirichlet_val_bottom]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

    location_fns = [left, right]

    problem = HeatConduction(mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    fwd_pred = ad_wrapper(problem)

    def k_eff_fn(rho, beta):

        rho = jnp.clip(rho, -1.0, 1.0).squeeze(-1)  # [64, 64]
        rho = (rho + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        # rho = jnp.clip(rho, 0.0, 1.0).squeeze(-1)  # [64, 64]
        rho_proj = (jnp.tanh(beta / 2) + jnp.tanh(beta * (rho - 0.5))) / (2 * jnp.tanh(beta / 2))
        k = k0 + rho_proj * (k1 - k0)

        k_vec = k.flatten(order='F').reshape((-1, 1))
        
        sol = fwd_pred(k_vec)

        T = sol[0].reshape((Ny + 1, Nx + 1), order='F')

        T_up = T[:-1, :-1]
        T_down = T[1:, :-1]

        dy = Ly / Ny
        
        q = k * (T_up - T_down) / dy
        q_avg = jnp.mean(q)
        k_eff = - q_avg * Ly / (T_top - T_bot)

        return k_eff

    def beta_fn(t):
        beta_min = 5.0
        beta_max = 10.0
        return beta_min * (beta_max / beta_min) ** t
    
    def s_fn(t):
        s_base = 2.0
        s_max = 50.0
        return jnp.array([s_base + (s_max - s_base) * (1.0 - t)], dtype=jnp.float64)
        

    BETA_MIN = 0.1
    BETA_MAX = 20.0
    
    def sde_beta(t):
        return BETA_MIN + t * (BETA_MAX - BETA_MIN)
    
    def sde_alpha_bar(t):
        integral_beta = BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2
        return jnp.exp(-0.5 * integral_beta)
    
    def sde_marginal_std(t):
        return jnp.sqrt(1.0 - sde_alpha_bar(t))


    key, init_key = random.split(key)
    x_t = random.normal(init_key, (1, IMG_SIZE, IMG_SIZE, 1), dtype=jnp.float64)

    eps_time = 1e-5
    timesteps = jnp.linspace(1.0, eps_time, NUM_STEPS + 1, dtype=jnp.float64)

    k_eff_history = []
    x_t_history = []
    
    print("\n" + "="*80)
    print("Starting SDE Sampling with Physics Guidance (float64 precision)")
    print(f"VP-SDE Parameters: beta_min={BETA_MIN}, beta_max={BETA_MAX}")
    print("="*80)

    for i in range(NUM_STEPS):
        
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_curr - t_next
        
        t_array = jnp.array([t_curr], dtype=jnp.float64)
        
        eps_pred = model.apply({'params': train_state.params}, x_t, t_array)
        
        mean_coef = jnp.sqrt(sde_alpha_bar(t_curr))
        std = sde_marginal_std(t_curr)
        
        mean_coef = jnp.maximum(mean_coef, 1e-6)
        std = jnp.maximum(std, 1e-6)
        
        x_0_pred = (x_t - std * eps_pred) / mean_coef
        
        beta = beta_fn(t_array)
        s = s_fn(t_curr)
        
        k_eff = k_eff_fn(x_0_pred, beta)
        
        def loss_fn_x0(x0):
            k = k_eff_fn(x0, beta)
            return (k - k_target) ** 2

        grad_loss_x0 = jax.grad(loss_fn_x0)(x_0_pred)
        
        grad_loss_xt = grad_loss_x0 / mean_coef
        
        score = -eps_pred / std
        
        grad_norm = jnp.linalg.norm(grad_loss_xt)
        grad_loss_xt_normalized = grad_loss_xt / (grad_norm + 1e-8)
        
        s = jnp.array([0.0]) if t_curr < 0.1 else s
        guided_score = score - s[0] * grad_loss_xt_normalized
        
        beta_t = sde_beta(t_curr)
        drift_coef = -0.5 * beta_t
        diffusion_coef = jnp.sqrt(beta_t)

        reverse_drift = drift_coef * x_t - beta_t * guided_score

        key, noise_key = random.split(key)
        z = random.normal(noise_key, x_t.shape, dtype=jnp.float64)
        
        x_t = x_t - reverse_drift * dt + diffusion_coef * jnp.sqrt(dt) * z
        
        k_eff_history.append(float(k_eff))
        x_t_history.append(np.array(x_0_pred))
        
        print(f"\nStep {i+1}/{NUM_STEPS} (t={float(t_curr):.4f}):")
        print(f"  k_eff (predicted x_0): {float(k_eff):.12f}")
        print(f"  |k_eff - k_target|: {float(jnp.abs(k_eff - k_target)):.6f}")
        print(f"  x_0_pred: mean={float(jnp.mean(x_0_pred)):.6f}, std={float(jnp.std(x_0_pred)):.6f}")
        print(f"  physics_grad_norm: {float(grad_norm):.6f}")
        print(f"  guidance_strength: {float(s[0]):.6f}")
    
    print("\n" + "="*80)
    print("SDE Sampling with Physics Guidance Completed")
    print("="*80)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_eff_history, label='k_eff', linewidth=2)
    plt.axhline(y=k_target, color='r', linestyle='--', label=f'Target k_eff = {k_target}', linewidth=2)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('k_eff', fontsize=12)
    plt.title('Evolution of Effective Thermal Conductivity - SDE Sampling (float64)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    Path("diffusion/samples").mkdir(exist_ok=True, parents=True)
    plt.savefig('diffusion/samples/k_eff_evolution_sde.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nk_eff evolution plot saved to diffusion/samples/k_eff_evolution_sde.png")



    
    np.save('diffusion/samples/k_eff_hstry_30-2.npy', k_eff_history)
    np.save('diffusion/samples/x_t_hstry_heat_30-2.npy', x_t_history)
    



    generated_img = x_t[0]
    generated_img = (generated_img + 1.0) / 2.0
    # generated_img = jnp.round(generated_img)
    generated_img = jnp.clip(generated_img, 0.0, 1.0)
    
    SAMPLE_SAVE_PATH = "diffusion/samples/k_eff_sample_guided.png"
    plt.imsave(SAMPLE_SAVE_PATH, np.array(generated_img).squeeze(), cmap='gray_r')

    print(f"\nFinal generated image saved as {SAMPLE_SAVE_PATH}")
    print(f"Final k_eff: {k_eff_history[-1]:.12f}")
    print(f"Target k_eff: {k_target:.12f}")
    print(f"Difference: {abs(k_eff_history[-1] - k_target):.12f}")
    
    print(f"\nData type verification:")
    print(f"  x_t dtype: {x_t.dtype}")

if __name__ == "__main__":
    main()