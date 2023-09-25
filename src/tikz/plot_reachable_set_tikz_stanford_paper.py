import pickle
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import numpy as np
import os

p_bins = np.linspace(-11, 11, 129, endpoint=True)
p_lbs = np.array(p_bins[:-1],dtype=np.float32)
p_ubs = np.array(p_bins[1:], dtype=np.float32)

theta_bins = np.linspace(-30, 30, 129, endpoint=True)
theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
theta_ubs = np.array(theta_bins[1:], dtype=np.float32)


fig, axs = plt.subplots(1, 5, figsize=(10, 2))
root_data_path = "./results/compute_reachable_sets/p_coeff_-0.74_theta_coeff_-0.44_p_lb_-10.0_p_ub_10.0_theta_lb_-10.0_theta_ub_10.0_p_range_lb_-11.0_p_range_ub_11.0_p_num_bin_128_theta_range_lb_-30.0_theta_range_ub_30.0_theta_num_bin_128"

# 0 s
ax = axs[0]
## read data 
data_path = os.path.join(root_data_path, "0s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
print(f"step 0, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}")

ax.scatter(simulations[:, 0], simulations[:, 1], s=0.05, color='black')

polygon = Polygon()

for idx, cell in enumerate(reachable_cells_baseline):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))

x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')
ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')

# 5 s
ax = axs[1]
## read data 
data_path = os.path.join(root_data_path, "5s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
print(f"step 5, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}")

ax.scatter(simulations[:, 0], simulations[:, 1], s=0.05, color='black')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_baseline):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_one_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='blue', alpha=0.5, label='_nolegend_')


ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')

# 10 s
ax = axs[2]
## read data 
data_path = os.path.join(root_data_path, "10s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
print(f"step 10, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}")

ax.scatter(simulations[:, 0], simulations[:, 1], s=0.05, color='black')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_baseline):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_one_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='blue', alpha=0.5, label='_nolegend_')


ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')

# 15 s
ax = axs[3]
## read data 
data_path = os.path.join(root_data_path, "15s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
print(f"step 15, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}")

ax.scatter(simulations[:, 0], simulations[:, 1], s=0.05, color='black')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_baseline):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_one_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='blue', alpha=0.5, label='_nolegend_')


ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')

# 20 s
ax = axs[4]
## read data 
data_path = os.path.join(root_data_path, "20s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
print(f"step 20, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}")

ax.scatter(simulations[:, 0], simulations[:, 1], s=0.05, color='black')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_baseline):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_one_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='blue', alpha=0.5, label='_nolegend_')


ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')

# converged
ax = axs[4]
## read data 
data_path = os.path.join(root_data_path, "converged", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
print(f"converged, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}")

ax.scatter(simulations[:, 0], simulations[:, 1], s=0.05, color='black')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_baseline):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_one_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='blue', alpha=0.5, label='_nolegend_')


ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')
plt.tight_layout()
import tikzplotlib
tikzplotlib.save("single_cell_reachable_set.tex")

plt.savefig("reachable_sets_stanford_paper.png")

