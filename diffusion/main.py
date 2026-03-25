import os
import numpy as np
import jax
import jax.numpy as jnp
import logging
import optax
import matplotlib.pyplot as plt
import flax.serialization
from flax.training.train_state import TrainState
from jax import random
from tqdm import tqdm

from utils import DataLoader, train_step, VPSDE
from utils import sample_sde as sample
from unet import UNet


jax.config.update("jax_default_matmul_precision", "float32")
# jax.config.update("jax_enable_x64", True)


def main():
    IMG_SIZE = 64
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 1000
    SAVE_EVERY_EPOCHS = 100
    DATA_DIR = "foam/data/64"
    LOG_FILE = "diffusion/training_errors.log"
    MODEL_DIR = "diffusion/models"
    SAMPLE_DIR = "diffusion/samples"
    
    BETA_MIN = 0.1
    BETA_MAX = 20.0
    
    NUM_SAMPLE_STEPS = 1000
    
    key = random.PRNGKey(42)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    dataloader = DataLoader(DATA_DIR, BATCH_SIZE, IMG_SIZE, max_samples=1)
    sde = VPSDE(beta_min=BETA_MIN, beta_max=BETA_MAX)

    model = UNet()
    
    dummy_x = jnp.ones((1, IMG_SIZE, IMG_SIZE, 1), dtype=jnp.float32)
    dummy_t = jnp.ones((1,), dtype=jnp.float32)

    key, model_key = random.split(key)
    params = model.init(model_key, dummy_x, dummy_t)['params']
    
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LEARNING_RATE, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    )

    train_state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    key, sample_key = random.split(key)
    generated_image = sample(
        sample_key, 
        num_steps=100,
        img_size=IMG_SIZE, 
        state=train_state, 
        model=model, 
        sde=sde
    )
    sample_path = f"{SAMPLE_DIR}/sample_epoch_0.png"
    plt.imsave(sample_path, np.array(generated_image).squeeze(), cmap='gray_r')
    
    
    for epoch in range(NUM_EPOCHS):
        key, epoch_key = random.split(key)
        
        epoch_losses = []
        pbar = tqdm(
            dataloader.__iter__(key=epoch_key), 
            total=len(dataloader), 
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        for batch in pbar:
            if jnp.isnan(batch).any() or jnp.isinf(batch).any():
                print(f"\n Batch NaN or Inf, skip...")
                continue
                
            key, train_key = random.split(key)
            train_state, loss, train_key = train_step(
                train_state, batch, train_key, model, sde
            )
 
            if jnp.isnan(loss) or jnp.isinf(loss):
                print(f"\n loss NaN or Inf, skip...")
                continue
            
            epoch_losses.append(float(loss))
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
            current_step = int(train_state.step)
            print(f"\n[Epoch {epoch + 1}, Step {current_step}] sampling...")
            
            key, sample_key = random.split(key)
            generated_image = sample(
                sample_key, 
                num_steps=NUM_SAMPLE_STEPS, 
                img_size=IMG_SIZE, 
                state=train_state, 
                model=model, 
                sde=sde
            )
            
            sample_path = f"{SAMPLE_DIR}/sample_epoch_{epoch+1}.png"
            plt.imsave(sample_path, np.array(generated_image).squeeze(), cmap='gray_r')
            print(f"sample saved as {sample_path}")
    
            params_bytes = flax.serialization.to_bytes(train_state.params)
            model_path = f"{MODEL_DIR}/vpsde_model_epoch_{epoch+1}.flax"
            with open(model_path, "wb") as f:
                f.write(params_bytes)
            print(f"model saved {model_path}")
    
    params_bytes = flax.serialization.to_bytes(train_state.params)
    final_model_path = f"{MODEL_DIR}/vpsde_model_final.flax"
    with open(final_model_path, "wb") as f:
        f.write(params_bytes)
    print(f"\n Done! model saved {final_model_path}")


if __name__ == "__main__":
    main()
