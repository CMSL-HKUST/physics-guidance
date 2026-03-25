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
from fem_pf import full_disps, fwd_pred_seq

#%%
# ----- initialization of unet -----
IMG_SIZE = 64
NUM_STEPS = 50
LEARNING_RATE = 1e-4
MODEL_PATH = "source/vpsde_model.flax"

grad_volfrac_flag = [True, False][1]
grad_eng_flag = [True, False][1]
target_volfrac = 0.4

# rng = int(time.time())
rng = 304
key = jax.random.PRNGKey(rng)
output_dir = os.path.join(os.path.dirname(__file__), f"diffusion/sample_{rng}_{target_volfrac}")
os.makedirs(output_dir, exist_ok=True)

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


k0 = 210e1
k1 = 210e3
theta_nu = 0.3 * np.ones((IMG_SIZE**2,1))
theta_G_c = 2.4
theta_l = 0.05

def eng_fn(rho, beta):

    rho_proj = (jnp.tanh(beta / 2) + jnp.tanh(beta * (rho - 0.5))) / (2 * jnp.tanh(beta / 2))
    E = k0 + rho_proj * (k1 - k0)
    theta_E = E[0,:,:,0][::-1, :].reshape(-1, 1, order='F')
    thetas = [theta_E, theta_nu, theta_G_c, theta_l]
    _, _, forces = fwd_pred_seq(thetas)
    eng = jnp.trapezoid(forces, full_disps)
    
    return eng

def compute_volfrac(rho, beta):
    rho_proj = (jnp.tanh(beta / 2) + jnp.tanh(beta * (rho - 0.5))) / (2 * jnp.tanh(beta / 2))
    volfrac = jnp.sum(rho_proj)/IMG_SIZE**2
    return volfrac

def beta_fn(t):
    beta_min = 5.0
    beta_max = 10.0
    return beta_min * (beta_max / beta_min) ** t

def s_fn(t):
    """
    linear intensity.
    """
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

eng_history = []
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
    
    volfrac = compute_volfrac(x_0_pred, beta) 
    
    def loss_fn_volfrac(x0):
        volfrac = compute_volfrac(x0, beta)
        return (volfrac* IMG_SIZE**2-target_volfrac* IMG_SIZE**2)**2 
    
    if grad_volfrac_flag:
        
        grad_volfrac = jax.grad(loss_fn_volfrac)(x_0_pred)
    
    else:
        grad_volfrac = 0.
    
    def loss_fn_x0(x0):
        eng = eng_fn(x0, beta)
        return -eng
    
    if grad_eng_flag:
        eng, grad_loss_x0 = jax.value_and_grad(loss_fn_x0)(x_0_pred)
        eng = -eng
    else:
        eng = 0.
        grad_loss_x0 = 0.
    
    grad_loss_x0 += grad_volfrac
    
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
    
    eng_history.append(float(eng))
    x_t_history.append(np.array(x_0_pred)) 
    
    print(f"\nStep {i+1}/{NUM_STEPS} (t={float(t_curr):.4f}):")
    print(f"  ENG: {float(eng):.12f}")
    print(f"  VOLFRAC: {float(volfrac):.12f}")
    print(f"  x_0_pred: mean={float(jnp.mean(x_0_pred)):.6f}, std={float(jnp.std(x_0_pred)):.6f}")
    print(f"  physics_grad_norm: {float(grad_norm):.6f}")
    print(f"  guidance_strength: {float(s[0]):.6f}")

print("\n" + "="*80)
print("SDE Sampling with Physics Guidance Completed")
print("="*80)

plt.figure(figsize=(10, 6))
plt.plot(eng_history, label='ENG', linewidth=2)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Iteration history', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir,'evolution_sde.png'), dpi=150, bbox_inches='tight')
plt.close()

np.save(os.path.join(output_dir, 'x_t_hstry_sde.npy'), x_t_history)

generated_img = x_t[0]
generated_img = (generated_img + 1.0) / 2.0
# generated_img = jnp.round(generated_img)
generated_img = jnp.clip(generated_img, 0.0, 1.0)

SAMPLE_SAVE_PATH = os.path.join(output_dir, "sample_guided_sde.png")
plt.imsave(SAMPLE_SAVE_PATH, np.array(generated_img).squeeze(), cmap='gray_r')

print(f"\nFinal generated image saved as {SAMPLE_SAVE_PATH}")

print(f"\nData type verification:")
print(f"  x_t dtype: {x_t.dtype}")


# if __name__ == "__main__":
#     main()

#%%
beta=beta_fn(timesteps[-2])
rho = np.load(os.path.join(output_dir, 'x_t_hstry_sde.npy'))[-1,...]
rho_proj = (jnp.tanh(beta / 2) + jnp.tanh(beta * (rho - 0.5))) / (2 * jnp.tanh(beta / 2))
E = k0 + rho_proj * (k1 - k0)
theta_E = E[0,:,:,0][::-1,:].reshape(-1, 1, order='F')
thetas = [theta_E, theta_nu, theta_G_c, theta_l]
sols_u, sols_d, forces = fwd_pred_seq(thetas)
np.savez(os.path.join(output_dir, 'sol.npz'), 
        sols_d = sols_d,
        sols_u = sols_u,
        forces = forces)
#%%
beta=beta_fn(timesteps[-2])
rho = np.load(os.path.join(output_dir, 'x_t_hstry_sde.npy'))[-1,...]
data = np.load(os.path.join(output_dir, 'sol.npz'))
forces = data['forces']
sols_u = data['sols_u']
sols_d = data['sols_d']
plt.figure(figsize=(10, 6))
plt.rc('font', size=20)
plt.plot(full_disps, forces, 'b--', linewidth=2, label='JAX-FEM')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Staggered scheme')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(compute_volfrac(rho, beta))
print(jnp.trapezoid(forces, full_disps))

from jax_fem_pf.utils import plot_quad_mesh
from fem_pf import mesh
plot_quad_mesh(mesh.points, mesh.cells, sols_d[-1])

