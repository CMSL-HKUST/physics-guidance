import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import time

def generate_2d_foam_intuitive(
    image_size=1024,          # 1.  image size 
    avg_pore_size=80,         # 2.  d_cell
    solid_fraction=0.30,      # 3.  solid fraction (0.0 to 1.0)
    roundness_factor=0.1,     # 4.  round factor
):


    num_points = int((image_size / avg_pore_size)**2)
    
    smoothing_sigma = avg_pore_size * roundness_factor
    
    initial_wall_thickness = max(2.0, smoothing_sigma * 0.5)

    points = np.random.rand(num_points, 2) * image_size
    kdtree = KDTree(points)
    
    y_coords, x_coords = np.indices((image_size, image_size))
    pixel_coords = np.c_[x_coords.ravel(), y_coords.ravel()]
    
    dists, _ = kdtree.query(pixel_coords, k=2)
    wall_mask = (dists[:, 1] - dists[:, 0]) < initial_wall_thickness
    
    initial_grid = np.zeros(image_size**2, dtype=np.float32)
    initial_grid[wall_mask] = 1.0
    initial_grid = initial_grid.reshape((image_size, image_size))

    smoothed_grid = gaussian_filter(initial_grid, sigma=smoothing_sigma)
    
    if solid_fraction <= 0.0:
        dynamic_threshold = smoothed_grid.max() + 1
    elif solid_fraction >= 1.0:
        dynamic_threshold = smoothed_grid.min() - 1
    else:
        quantile_val = 1.0 - solid_fraction
        dynamic_threshold = np.quantile(smoothed_grid, quantile_val)
    
    final_image = (smoothed_grid > dynamic_threshold).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(final_image, cmap='gray_r')
    ax.axis('off')
    plt.show()
    
    # title = (
    #     f"2D Foam (Intuitive Control)\n"
    #     f"Avg Pore Size: {avg_pore_size}px, Solid Fraction: {solid_fraction*100:.1f}%, "
    #     f"Roundness: {roundness_factor}"
    # )
    # plt.title(title)
    # plt.tight_layout()        
    # plt.show()

    return final_image

if __name__ == '__main__':
    generate_2d_foam_intuitive(
            image_size=64,
            avg_pore_size=20, 
            solid_fraction=0.4, 
            roundness_factor=0.1
        )