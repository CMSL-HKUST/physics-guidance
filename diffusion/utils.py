import os
import numpy as np
import glob
import math
import cv2
import jax
import jax.numpy as jnp
from jax import random
from functools import partial


jax.config.update("jax_default_matmul_precision", "float32")
# jax.config.update("jax_enable_x64", True)


# ==========================================
# VP-SDE
# ==========================================
class VPSDE:
    """
    Variance Preserving SDE (VP-SDE)
    
    Forward SDE: dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
    
    beta(t) = beta_min + t * (beta_max - beta_min)
    
        x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * eps
        
        alpha(t) = 1 - beta(t)
        alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds)
    """
    
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta(self, t):
        """beta(t) = beta_min + t * (beta_max - beta_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def integral_beta(self, t):
        """integral_0^t beta(s) ds"""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
    
    def alpha_bar(self, t):
        """
        alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds)
        """
        return jnp.exp(-0.5 * self.integral_beta(t))
    
    def marginal_prob(self, x_0, t):
        """
        p(x_t | x_0)
        
        x_t = mean_coef * x_0 + std * eps, eps ~ N(0, I)
        
        Returns:
            mean_coef: sqrt(alpha_bar(t))
            std: sqrt(1 - alpha_bar(t))
        """
        alpha_bar_t = self.alpha_bar(t)
        mean_coef = jnp.sqrt(alpha_bar_t)
        std = jnp.sqrt(1.0 - alpha_bar_t)
        return mean_coef, std
    
    def prior_sampling(self, key, shape):
        return random.normal(key, shape)
    
    def sde_coefficients(self, t):
        """
        dx = f(x,t) dt + g(t) dw
        
        VP-SDE:
            f(x, t) = -0.5 * beta(t) * x  (drift)
            g(t) = sqrt(beta(t))          (diffusion)
        
        Returns:
            drift_coef: -0.5 * beta(t)
            diffusion: sqrt(beta(t))
        """
        beta_t = self.beta(t)
        drift_coef = -0.5 * beta_t
        diffusion = jnp.sqrt(beta_t)
        return drift_coef, diffusion
    
    def reverse_sde_coefficients(self, x, t, score):
        """
        dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dw_reverse
        
        score = ∇_x log p_t(x)
        
        VP-SDE:
            f(x, t) = -0.5 * beta(t) * x
            g(t) = sqrt(beta(t))
        
        Returns:
            drift: f(x,t) - g(t)^2 * score
            diffusion: g(t)
        """
        beta_t = self.beta(t)
        drift_coef = -0.5 * beta_t
        diffusion = jnp.sqrt(beta_t)
        
        reverse_drift = drift_coef * x - beta_t * score
        return reverse_drift, diffusion


class DataLoader:
    def __init__(self, data_dir, batch_size, img_size, max_samples=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.file_paths = glob.glob(os.path.join(self.data_dir, "*.npy"))
        if not self.file_paths:
            raise ValueError(f"no npy. file in '{self.data_dir}'")
        
        if max_samples is not None and max_samples < len(self.file_paths):
            np.random.shuffle(self.file_paths)
            self.file_paths = self.file_paths[:max_samples]
    
    def __len__(self):
        return math.ceil(len(self.file_paths) / self.batch_size)
    
    def __iter__(self, key=None):
        file_paths = self.file_paths.copy()
        if key is None:
            np.random.shuffle(file_paths)
        else:
            np_seed = int(key[0]) if isinstance(key, jnp.ndarray) else int(key)
            rng = np.random.default_rng(np_seed)
            rng.shuffle(file_paths)
        num_skipped = 0
        for i in range(0, len(file_paths), self.batch_size):
            batch_paths = file_paths[i:i+self.batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    img = np.load(path).astype(np.float32)
                    # Check for NaN or Inf in loaded data
                    if np.isnan(img).any() or np.isinf(img).any():
                        print(f"warning: file {path} has NaN or Inf, skip.")
                        num_skipped += 1
                        continue
                    if img.max() > 1.0:
                        img = img / 255.0
                    # Clip to valid range
                    img = np.clip(img, 0.0, 1.0)
                    if img.shape != (self.img_size, self.img_size):
                        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                    img = np.expand_dims(img, axis=-1)
                    img = img * 2.0 - 1.0
                    batch_images.append(img)
                except Exception as e:
                    print(f"warning: error when load {path}: {e}, skip.")
                    num_skipped += 1
            if not batch_images:
                continue
            yield jnp.array(batch_images)
        if num_skipped > 0:
            print(f" {num_skipped} files skipped.")


# ==========================================
# VP-SDE training
# ==========================================
@partial(jax.jit, static_argnums=(3, 4))
def train_step(state, batch, key, model, sde):

    key, noise_key, t_key = random.split(key, 3)
    batch_size = batch.shape[0]
    
    eps_time = 1e-5
    t = random.uniform(t_key, (batch_size,), minval=eps_time, maxval=1.0)
    
    eps = random.normal(noise_key, batch.shape)

    mean_coef, std = sde.marginal_prob(batch, t)
    mean_coef = mean_coef[:, None, None, None]
    std = std[:, None, None, None]
    x_t = mean_coef * batch + std * eps
    
    def loss_fn(params):
        eps_pred = model.apply({'params': params}, x_t, t)
        loss = jnp.mean((eps_pred - eps) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads
    )
    
    state = state.apply_gradients(grads=grads)
    return state, loss, key


# ==========================================
# VP-SDE sampling
# ==========================================
@partial(jax.jit, static_argnums=(1, 2, 4, 5))
def sample_sde(key, num_steps, img_size, state, model, sde):
    
    key, init_key = random.split(key)
    x_t = sde.prior_sampling(init_key, (1, img_size, img_size, 1))
    
    eps_time = 1e-5
    timesteps = jnp.linspace(1.0, eps_time, num_steps + 1)
    
    def sde_step(carry, i):
        x_t, key = carry
        key, noise_key = random.split(key)
        
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_curr - t_next
        
        t_array = jnp.array([t_curr])
        eps_pred = model.apply({'params': state.params}, x_t, t_array)
        
        _, std = sde.marginal_prob(x_t, t_curr)

        std = jnp.maximum(std, 1e-6)
        score = -eps_pred / std
        
        beta_t = sde.beta(t_curr)
        drift_coef = -0.5 * beta_t
        diffusion = jnp.sqrt(beta_t)
        
        reverse_drift = drift_coef * x_t - beta_t * score
        
        z = random.normal(noise_key, x_t.shape)
        
        noise_scale = jnp.where(i < num_steps - 1, 1.0, 0.0)
        x_next = x_t - reverse_drift * dt + noise_scale * diffusion * jnp.sqrt(dt) * z
        
        return (x_next, key), None
    
    (x_0, _), _ = jax.lax.scan(sde_step, (x_t, key), jnp.arange(num_steps))
    
    generated_img = x_0[0]
    generated_img = (generated_img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    generated_img = jnp.clip(generated_img, 0.0, 1.0)
    return generated_img




