import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax import config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config.update("jax_enable_x64", True)
# =============================

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
from scipy.ndimage import uniform_filter1d
from unet import UNet


from mech.mpm_mls_grad_v3 import init_particles_from_density, run_simulation_with_curve


def main():

    # ----- initialization of unet -----
    IMG_SIZE = 64
    NUM_STEPS = 50
    LEARNING_RATE = 1e-4
    MODEL_PATH = "diffusion/models/vpsde_model.flax"
    
    START_GUIDANCE_STEP = 50
    
    # key = jax.random.PRNGKey(int(time.time()))
    key = jax.random.PRNGKey(12)

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

    reference_voxel = jnp.load('diffusion/samples/print/p0/voxel_k12.npy')

    grid_padding_cells = 20
    n_grid = max(reference_voxel.shape) + 2 * grid_padding_cells
    dx_mpm = 1.0 / n_grid
    inv_dx = float(n_grid)

    p_x_ref, _, _, _, _, _, _, _, _ = init_particles_from_density(reference_voxel, dx_mpm)
    initial_y_min = float(jnp.min(p_x_ref[:, 1]))
    initial_y_max = float(jnp.max(p_x_ref[:, 1]))
    initial_x_min = float(jnp.min(p_x_ref[:, 0]))
    initial_x_max = float(jnp.max(p_x_ref[:, 0]))
    
    actual_initial_height = initial_y_max - initial_y_min
    ground_y_pos = initial_y_min
    initial_y_top = initial_y_max
    contact_area = initial_x_max - initial_x_min
    
    n_steps = 10000
    
    stress_record_interval = 100 
    
    target = jnp.load('diffusion/target_4.npy')
    target_strain_curve, target_stress_curve = target[1:, 0], target[1:, 1]
    n_curve_points = len(target_stress_curve)
    
    def compute_stress_curve(rho, beta=None):

        if rho.ndim == 4:
            rho = rho[0]  # [64, 64, 1]
        
        # ===== [0, 1] =====
        rho_normalized = (rho + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        voxel_density = jnp.clip(rho_normalized, 0.0, 1.0).squeeze(-1)  # [64, 64]

        voxel_density = jnp.pad(
        voxel_density,
        pad_width=((2, 2), (0, 0)), 
        mode='constant',
        constant_values=1.0
        )
        
        try:
            stress_curve, _ = run_simulation_with_curve(
                voxel_density, n_steps, n_grid, inv_dx, 
                initial_y_top, ground_y_pos,
                checkpoint_every=stress_record_interval
            )
        except Exception as e:
            print(f"  WARNING: run_simulation_with_curve failed: {e}")
            stress_curve = jnp.zeros(n_curve_points, dtype=jnp.float64)
        
        stress_curve = jnp.nan_to_num(stress_curve, nan=0.0, posinf=0.0, neginf=0.0)
        
        return stress_curve

    def beta_fn(t):
        beta_min = 5.0
        beta_max = 10.0
        return beta_min * (beta_max / beta_min) ** t
    
    def s_fn(t):
        s_base = 0.0
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

    curve_mse_history = []
    x_t_history = []
    
    print("\n" + "="*80)
    print("Starting SDE Sampling with Physics Guidance (float64 precision)")
    print(f"VP-SDE Parameters: beta_min={BETA_MIN}, beta_max={BETA_MAX}")
    print("="*80)
    
    for i in range(NUM_STEPS):

        start_time = time.time()
        
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
        
        x_0_pred = jnp.clip(x_0_pred, -1.0, 1.0)

        has_nan_or_inf = jnp.any(jnp.isnan(x_0_pred)) | jnp.any(jnp.isinf(x_0_pred))
        
        beta = beta_fn(t_array)
        s = s_fn(t_curr)
        
        should_compute_guidance = (i + 1 > START_GUIDANCE_STEP and 
                                   not has_nan_or_inf and 
                                   t_curr >= 0.10) 
        
        if should_compute_guidance:
            def loss_fn_x0(x0):
                pred_stress = compute_stress_curve(x0)
                stress_mse = jnp.mean((pred_stress - target_stress_curve) ** 2)
                return stress_mse, pred_stress
            
            (curve_loss, pred_stress_curve), grad_loss_x0 = jax.value_and_grad(
                loss_fn_x0, has_aux=True
            )(x_0_pred)
            
            stress_max = jnp.max(jnp.abs(pred_stress_curve))
            stress_has_issues = (jnp.any(jnp.isnan(pred_stress_curve)) | 
                                jnp.any(jnp.isinf(pred_stress_curve)) | 
                                (stress_max > 1e10))
            
            if stress_has_issues:
                print(f"  WARNING: Stress curve has issues (max={float(stress_max):.2e}), skipping guidance")
                grad_loss_xt = jnp.zeros_like(x_0_pred)
                curve_loss = float('inf')
            else:
                grad_loss_xt = grad_loss_x0 / mean_coef
                
                grad_has_nan = jnp.any(jnp.isnan(grad_loss_xt)) | jnp.any(jnp.isinf(grad_loss_xt))
                if grad_has_nan:
                    print(f"  WARNING: Gradient contains NaN/Inf, skipping guidance")
                    grad_loss_xt = jnp.zeros_like(x_0_pred)
                    curve_loss = float('inf')
                else:
                    grad_loss_xt = jnp.clip(grad_loss_xt, -100.0, 100.0)
        else:
            pred_stress_curve = jnp.zeros(n_curve_points, dtype=jnp.float64)
            curve_loss = 0.0
            grad_loss_xt = jnp.zeros_like(x_0_pred)
            if has_nan_or_inf:
                print(f"  WARNING: x_0_pred contains NaN/Inf, skipping guidance")
        
        score = -eps_pred / std
        
        grad_norm = jnp.linalg.norm(grad_loss_xt)
        grad_loss_xt_normalized = grad_loss_xt / (grad_norm + 1e-8)
        
        s = jnp.array([0.0]) if t_curr < 0.10 else s
        guided_score = score - s[0] * grad_loss_xt_normalized
        
        beta_t = sde_beta(t_curr)
        drift_coef = -0.5 * beta_t
        diffusion_coef = jnp.sqrt(beta_t)

        reverse_drift = drift_coef * x_t - beta_t * guided_score

        key, noise_key = random.split(key)
        z = random.normal(noise_key, x_t.shape, dtype=jnp.float64)
        
        x_t = x_t - reverse_drift * dt + diffusion_coef * jnp.sqrt(dt) * z

        x_t.block_until_ready()
        end_time = time.time()
        step_time = end_time - start_time

        curve_mse_history.append(float(curve_loss))
        x_t_history.append(np.array(x_0_pred))
        
        print(f"\nStep {i+1}/{NUM_STEPS} (t={float(t_curr):.4f}, Step Time: {step_time:.3f} seconds):")
        print(f"  x_0_pred: mean={float(jnp.mean(x_0_pred)):.6f}, std={float(jnp.std(x_0_pred)):.6f}, range=[{float(jnp.min(x_0_pred)):.2f}, {float(jnp.max(x_0_pred)):.2f}]")
        if should_compute_guidance:
            print(f"  [Physics Guidance Active]")
            print(f"  curve_loss (MSE): {float(curve_loss):.6e}")
            print(f"  pred_stress range: [{float(jnp.min(pred_stress_curve)):.2e}, {float(jnp.max(pred_stress_curve)):.2e}] Pa")
            print(f"  target_stress range: [{float(jnp.min(target_stress_curve)):.2e}, {float(jnp.max(target_stress_curve)):.2e}] Pa")
            print(f"  physics_grad_norm: {float(grad_norm):.6f}")
            print(f"  guidance_strength: {float(s[0]):.6f}")
        elif i + 1 <= START_GUIDANCE_STEP:
            print(f"  [No Physics Guidance - Structure Formation Phase]")
        elif t_curr < 0.10:
            print(f"  [No Physics Guidance - Final Denoising Phase]")
        elif has_nan_or_inf:
            print(f"  [Physics Guidance Skipped - Numerical Issues]")
        else:
            print(f"  [No Physics Guidance - Time threshold not met (t={float(t_curr):.4f})]")
        
        if (i + 1) % 5 == 0:
            print(f"  [Saving checkpoint at step {i+1}]")

            x_0_normalized = (x_0_pred + 1.0) / 2.0
            x_0_normalized = jnp.clip(x_0_normalized, 0.0, 1.0)
            
            checkpoint_dir = Path("diffusion/samples/checkpoints")
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            voxel_path = checkpoint_dir / f"voxel_step_{i+1:03d}.npy"
            np.save(voxel_path, np.array(x_0_normalized.squeeze()))

            try:
                voxel_for_sim = x_0_normalized.squeeze()
                if voxel_for_sim.ndim != 2:
                    raise ValueError(f"Expected 2D voxel, got shape {voxel_for_sim.shape}")
                
                current_stress, current_strain = run_simulation_with_curve(
                    voxel_for_sim, n_steps, n_grid, inv_dx,
                    initial_y_top, ground_y_pos,
                    checkpoint_every=stress_record_interval
                )

                record = np.stack([current_stress, current_strain], axis=1)
                record_path = checkpoint_dir / f"curve_step_{i+1:03d}.npy"
                np.save(record_path, record)




                
                current_stress = np.array(current_stress)
                current_stress = uniform_filter1d(current_stress, size=3).tolist()

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                ax1 = axes[0]
                ax1.imshow(np.array(x_0_normalized.squeeze()), cmap='gray_r')
                ax1.set_title(f'Generated Foam - Step {i+1}/{NUM_STEPS}', fontsize=24)
                ax1.axis('off')
                
                ax2 = axes[1]
                ax2.plot(np.array(target_strain_curve), np.array(target_stress_curve),
                            'r-', linewidth=2.5, label='Target', alpha=0.8, marker='o', markersize=3)
                ax2.plot(np.array(current_strain), np.array(current_stress),
                            'b--', linewidth=2.5, label='Current', alpha=0.8, marker='s', markersize=3)
                ax2.set_xlabel('Displacement (mm)', fontsize=20)
                ax2.set_ylabel('Load (N)', fontsize=20)
                ax2.tick_params(axis='both', labelsize=18)
                ax2.set_title(f'Load-Displacement Curve - Step {i+1}/{NUM_STEPS}', fontsize=24)
                ax2.legend(fontsize=20)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                fig_path = checkpoint_dir / f"fig_step_{i+1:03d}.png"
                plt.savefig(fig_path, dpi=120, bbox_inches='tight')
                plt.close()
                
                print(f"    Saved: {voxel_path.name}, {record_path.name}, {fig_path.name}")
            except Exception as e:
                print(f"    Warning: Failed to compute/plot curve: {e}")
                
    
    print("\n" + "="*80)
    print("SDE Sampling with Physics Guidance Completed")
    print("="*80)

    print("\nComputing final stress-strain curve for visualization...")

    final_x_normalized = (x_t[0] + 1.0) / 2.0
    final_x_normalized = jnp.clip(final_x_normalized, 0.0, 1.0).squeeze(-1)
    
    final_stress_curve, final_strain_curve = run_simulation_with_curve(
        final_x_normalized, n_steps, n_grid, inv_dx,
        initial_y_top, ground_y_pos,
        checkpoint_every=stress_record_interval
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    
    ax1 = axes[0]
    ax1.plot(curve_mse_history, label='Curve MSE Loss', linewidth=2, color='blue')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Evolution of Stress Curve MSE Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    # ax1.set_yscale('log')

    ax2 = axes[1]
    ax2.plot(np.array(target_strain_curve), np.array(target_stress_curve),
             'r-', linewidth=2.5, label='Target', alpha=0.8, marker='o', markersize=4)
    ax2.plot(np.array(final_strain_curve), np.array(final_stress_curve),
             'b--', linewidth=2.5, label='Generated', alpha=0.8, marker='s', markersize=4)
    ax2.set_xlabel('Strain', fontsize=12)
    ax2.set_ylabel('Stress (Pa)', fontsize=12)
    ax2.set_title('Stress-Strain Curve Comparison', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path("diffusion/samples").mkdir(exist_ok=True, parents=True)
    plt.savefig('diffusion/samples/curve_matching_evolution_mpm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nCurve matching evolution plot saved to diffusion/samples/curve_matching_evolution_mpm.png")
    
    np.save('diffusion/samples/x_t_hstry.npy', x_t_history)
    np.save('diffusion/samples/curve_mse_hstry.npy', curve_mse_history)
    
    generated_img = x_t[0]
    generated_img = (generated_img + 1.0) / 2.0
    # generated_img = jnp.round(generated_img)
    generated_img = jnp.clip(generated_img, 0.0, 1.0)
    
    SAMPLE_SAVE_PATH = "diffusion/samples/sample_guided_mpm.png"
    plt.imsave(SAMPLE_SAVE_PATH, np.array(generated_img).squeeze(), cmap='gray_r')

    print(f"\nFinal generated image saved as {SAMPLE_SAVE_PATH}")
    print(f"Final curve MSE loss: {curve_mse_history[-1]:.6e}")
    
    print(f"\nFinal Results:")
    print(f"  Generated stress range: [{float(jnp.min(final_stress_curve)):.2e}, {float(jnp.max(final_stress_curve)):.2e}] Pa")
    print(f"  Target stress range:    [{float(jnp.min(target_stress_curve)):.2e}, {float(jnp.max(target_stress_curve)):.2e}] Pa")
    print(f"  Generated strain range: [{float(jnp.min(final_strain_curve)):.4f}, {float(jnp.max(final_strain_curve)):.4f}]")
    print(f"  Target strain range:    [{float(jnp.min(target_strain_curve)):.4f}, {float(jnp.max(target_strain_curve)):.4f}]")
    
    stress_rmse = float(jnp.sqrt(jnp.mean((final_stress_curve - target_stress_curve) ** 2)))
    stress_mae = float(jnp.mean(jnp.abs(final_stress_curve - target_stress_curve)))
    max_abs_error = float(jnp.max(jnp.abs(final_stress_curve - target_stress_curve)))
    
    target_stress_mean = float(jnp.mean(target_stress_curve))
    relative_rmse = stress_rmse / target_stress_mean * 100 if target_stress_mean > 0 else 0
    
    print(f"\nCurve Matching Metrics:")
    print(f"  RMSE: {stress_rmse:.6e} Pa ({relative_rmse:.2f}% of mean target stress)")
    print(f"  MAE:  {stress_mae:.6e} Pa")
    print(f"  Max absolute error: {max_abs_error:.6e} Pa")
    
    print(f"\nData type verification:")
    print(f"  x_t dtype: {x_t.dtype}")

if __name__ == "__main__":
    main()