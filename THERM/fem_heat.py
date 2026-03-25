import os
import numpy as np
import jax
import jax.numpy as jnp
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
import matplotlib.pyplot as plt


class HeatConduction(Problem):
    def __init__(self, *args, b=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.b = b

    def custom_init(self):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def conductivity(gradT, theta_k):
            return theta_k * gradT
        return conductivity
    
    def get_mass_map(self):
        def mass_map(u, x, theta_k):
            return -jnp.array([self.b])
        return mass_map
    
    def get_surface_maps(self):
        def surface_map(u, x):
            return -jnp.array([0])
        return [surface_map, surface_map]
    
    def set_params(self, params):
        full_params = jnp.ones((self.fes[0].num_cells, params.shape[1]))
        full_params = full_params.at[self.fes[0].flex_inds].set(params)
        thetas = jnp.repeat(full_params[:, None, :], self.fes[0].num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]


# --- k_eff ---
def k_eff_fn(rho, k0, k1, dy, T_top, T_bot, beta, fn):

    rho_proj = (jnp.tanh(beta / 2) + jnp.tanh(beta * (rho - 0.5))) / (2 * jnp.tanh(beta / 2))
    k = k0 + rho_proj * (k1 - k0)
    k_vec = k.flatten(order='F').reshape((-1, 1))
    
    sol = fn(k_vec)
    
    T = sol[0].reshape((Ny + 1, Nx + 1), order='F') 
    T_up = T[:-1, :-1]
    T_down = T[1:, :-1]
    
    q = k * (T_up - T_down) / dy
    q_avg = jnp.mean(q)
    k_eff = - q_avg * Ly / (T_top - T_bot) # keep the k positive

    return k_eff

if __name__ == '__main__':
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.
    Nx, Ny = 64, 64
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    try:
        foam_np = np.load('therm/2d_voxel.npy')
    except FileNotFoundError:
        print("no .npy file")
        x, y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
        foam_np = (np.sin(x * 10 * np.pi) * np.sin(y * 10 * np.pi) > 0).astype(int)

    foam = jnp.array(foam_np).astype(jnp.float32)

    # --- boundary conditions ---
    def left(point): return jnp.isclose(point[0], 0., atol=1e-5)
    def right(point): return jnp.isclose(point[0], Lx, atol=1e-5)
    def bottom(point): return jnp.isclose(point[1], 0., atol=1e-5)
    def top(point): return jnp.isclose(point[1], Ly, atol=1e-5)

    def dirichlet_val_top(point): return 100.
    def dirichlet_val_bottom(point): return 0.

    location_fns_dirichlet = [top, bottom]
    value_fns = [dirichlet_val_top, dirichlet_val_bottom]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

    location_fns = [left, right]

    problem = HeatConduction(mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    fwd_pred = ad_wrapper(problem)

    T_top = 100.
    T_bot = 0.
    rho_input = foam.astype(jnp.float32)
    k_eff_value = k_eff_fn(rho_input, k0=0.1, k1=100., T_top=100., T_bot=0., fn=fwd_pred)
    print(f"k_eff: {k_eff_value}")

    grad_fn = jax.grad(k_eff_fn)
    grad_val = grad_fn(rho_input, k0=0.1, k1=100., T_top=100., T_bot=0., fn=fwd_pred)

    print(f"grad shape: {grad_val.shape}")
    print(f"grad max: {jnp.max(jnp.abs(grad_val))}")

    