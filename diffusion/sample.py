import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.serialization
import matplotlib.pyplot as plt
from jax import random
from flax.training.train_state import TrainState
import os

from utils import VPSDE, sample_sde
from unet import UNet

def sample_sde_with_save(key, num_steps, img_size, state, model, sde, save_dir, save_every=10):

    os.makedirs(save_dir, exist_ok=True)
    
    # init: x_T ~ N(0, I)
    key, init_key = random.split(key)
    x_t = sde.prior_sampling(init_key, (1, img_size, img_size, 1))

    x_display = (x_t[0] + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    x_display = jnp.clip(x_display, 0.0, 1.0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.array(x_display).squeeze(), cmap='gray_r')
    ax.axis('off')
    save_path = os.path.join(save_dir, f'step_0.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # t: 1 -> 0
    eps_time = 1e-5
    timesteps = jnp.linspace(1.0, eps_time, num_steps + 1)
    
    for i in range(num_steps):
        key, noise_key = random.split(key)
        
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_curr - t_next
        
        # score = -eps_pred / std
        t_array = jnp.array([t_curr])
        eps_pred = model.apply({'params': state.params}, x_t, t_array)
        
        _, std = sde.marginal_prob(x_t, t_curr)
        std = jnp.maximum(std, 1e-6)
        score = -eps_pred / std
        
        # SDE coefficients
        beta_t = sde.beta(t_curr)
        drift_coef = -0.5 * beta_t
        diffusion = jnp.sqrt(beta_t)
        
        # reverse drift
        reverse_drift = drift_coef * x_t - beta_t * score
        
        # Euler-Maruyama
        z = random.normal(noise_key, x_t.shape)
        noise_scale = 1.0 if i < num_steps - 1 else 0.0
        x_t = x_t - reverse_drift * dt + noise_scale * diffusion * jnp.sqrt(dt) * z
        
        # save_every
        if (i + 1) % save_every == 0 or i == num_steps - 1:
            x_display = (x_t[0] + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            x_display = jnp.clip(x_display, 0.0, 1.0)

            # x_display = x_t[0]
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(np.array(x_display).squeeze(), cmap='gray_r')
            ax.axis('off')
            save_path = os.path.join(save_dir, f'step_{i+1:03d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"Saved step {i+1}/{num_steps} to {save_path}")
    
    generated_img = x_t[0]
    generated_img = (generated_img + 1.0) / 2.0
    generated_img = jnp.clip(generated_img, 0.0, 1.0)
    return generated_img

def main():
    IMG_SIZE = 64
    NUM_STEPS = 50
    LEARNING_RATE = 1e-4
    MODEL_PATH = "diffusion/models/vpsde_model.flax"
    SAVE_DIR = "diffusion/samples/sampling_steps"
    sde = VPSDE(beta_min=0.1, beta_max=20.0, T=1.0)

    key = jax.random.PRNGKey(34)

    model = UNet(sample_size=IMG_SIZE)

    dummy_x = jnp.ones((1, IMG_SIZE, IMG_SIZE, 1))
    dummy_t = jnp.ones((1,))
    key, model_key = random.split(key)
    params = model.init(model_key, dummy_x, dummy_t)['params']
    optimizer = optax.adam(LEARNING_RATE)

    with open(MODEL_PATH, "rb") as f:
        params_bytes = f.read()
    restored_params = flax.serialization.from_bytes(params, params_bytes)
    print("model parameters loaded.")

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=restored_params,
        tx=optimizer
    )

    key, sample_key = random.split(key)
    generated_image = sample_sde_with_save(
        sample_key, 
        num_steps=NUM_STEPS, 
        img_size=IMG_SIZE, 
        state=train_state, 
        model=model,
        sde=sde,
        save_dir=SAVE_DIR,
        save_every=10
    )

    final_path = "diffusion/samples/sample_final.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.array(generated_image).squeeze(), cmap='gray_r')
    ax.axis('off')
    plt.savefig(final_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"\nFinal image saved as {final_path}")

if __name__ == "__main__":
    main()