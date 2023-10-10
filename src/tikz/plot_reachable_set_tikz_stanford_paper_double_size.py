import pickle
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import numpy as np
import os

p_bins = np.linspace(-11, 11, 65, endpoint=True)
p_lbs = np.array(p_bins[:-1],dtype=np.float32)
p_ubs = np.array(p_bins[1:], dtype=np.float32)

theta_bins = np.linspace(-30, 30, 65, endpoint=True)
theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
theta_ubs = np.array(theta_bins[1:], dtype=np.float32)


fig, axs = plt.subplots(1, 1, figsize=(5, 5))
root_data_path = "./results/compute_reachable_sets/p_coeff_-0.74_theta_coeff_-0.44_p_lb_-9.0_p_ub_9.0_theta_lb_-10.0_theta_ub_10.0_p_range_lb_-11.0_p_range_ub_11.0_p_num_bin_64_theta_range_lb_-30.0_theta_range_ub_30.0_theta_num_bin_64"

# 0 s
ax = axs
## read data 
data_path = os.path.join(root_data_path, "0s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.round(np.concatenate([simulation_random, simulation_eager], axis=0), 2)
p_simulations = [round(i, 2) for i in list(simulations[:, 0])]
theta_simulations = [round(i, 2) for i in list(simulations[:, 1])]
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
reachable_cells_two_step_method = data["reachable_cells_two_step_method"]
print(f"step 0, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}, cells of two step method: {len(reachable_cells_two_step_method)}")

ax.scatter(p_simulations, theta_simulations, s=0.05, color='black')

polygon = Polygon()

for idx, cell in enumerate(reachable_cells_baseline):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))

x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

#rec = plt.Rectangle((-11, -30), 1, 60, color='pink', alpha=0.2)
#ax.add_patch(rec)
#rec = plt.Rectangle((10, -30), 1, 60, color='pink', alpha=0.2)
#ax.add_patch(rec)
unsafe_polygon = Polygon()
unsafe_patch_1 = [-11, -10, -30, 30]
unsafe_patch_2 = [10, 11, -30, 30]
unsafe_polygon1 = unsafe_polygon.union(Polygon([[unsafe_patch_1[0], unsafe_patch_1[2]], [unsafe_patch_1[0], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[2]]]))
x, y = unsafe_polygon1.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')
unsafe_polygon2 = unsafe_polygon.union(Polygon([[unsafe_patch_2[0], unsafe_patch_2[2]], [unsafe_patch_2[0], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[2]]]))
x, y = unsafe_polygon2.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')

ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')


import tikzplotlib
tikzplotlib.save("single_cell_reachable_set_0s.tex")

plt.savefig("reachable_sets_stanford_paper_0s.png")
plt.close()

# 3 s
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
ax = axs
## read data 
data_path = os.path.join(root_data_path, "3s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.round(np.concatenate([simulation_random, simulation_eager], axis=0), 2)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
reachable_cells_two_step_method = data["reachable_cells_two_step_method"]
print(f"step 3, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}, cells of two step method: {len(reachable_cells_two_step_method)}")

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

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_two_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='yellow', alpha=0.5, label='_nolegend_')


#rec = plt.Rectangle((-11, -30), 1, 60, color='pink', alpha=0.2)
#ax.add_patch(rec)
#rec = plt.Rectangle((10, -30), 1, 60, color='pink', alpha=0.2)
#ax.add_patch(rec)

unsafe_polygon = Polygon()
unsafe_patch_1 = [-11, -10, -30, 30]
unsafe_patch_2 = [10, 11, -30, 30]
unsafe_polygon1 = unsafe_polygon.union(Polygon([[unsafe_patch_1[0], unsafe_patch_1[2]], [unsafe_patch_1[0], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[2]]]))
x, y = unsafe_polygon1.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')
unsafe_polygon2 = unsafe_polygon.union(Polygon([[unsafe_patch_2[0], unsafe_patch_2[2]], [unsafe_patch_2[0], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[2]]]))
x, y = unsafe_polygon2.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')
ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')

tikzplotlib.save("single_cell_reachable_set_3s.tex")

plt.savefig("reachable_sets_stanford_paper_3s.png")
plt.close()

# 10 s
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
ax = axs
## read data 
data_path = os.path.join(root_data_path, "10s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
simulations = np.round(np.concatenate([simulation_random, simulation_eager], axis=0), 2)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
reachable_cells_two_step_method = data["reachable_cells_two_step_method"]
print(f"step 10, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}, cells of two step method: {len(reachable_cells_two_step_method)}")

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

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_two_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='yellow', alpha=0.5, label='_nolegend_')

#rec = plt.Rectangle((-11, -30), 1, 60, color='pink', alpha=0.2)
#ax.add_patch(rec)
#rec = plt.Rectangle((10, -30), 1, 60, color='pink', alpha=0.2)
#ax.add_patch(rec)
unsafe_polygon = Polygon()
unsafe_patch_1 = [-11, -10, -30, 30]
unsafe_patch_2 = [10, 11, -30, 30]
unsafe_polygon1 = unsafe_polygon.union(Polygon([[unsafe_patch_1[0], unsafe_patch_1[2]], [unsafe_patch_1[0], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[2]]]))
x, y = unsafe_polygon1.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')
unsafe_polygon2 = unsafe_polygon.union(Polygon([[unsafe_patch_2[0], unsafe_patch_2[2]], [unsafe_patch_2[0], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[2]]]))
x, y = unsafe_polygon2.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')

ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')
tikzplotlib.save("single_cell_reachable_set_10s.tex")
plt.savefig("reachable_sets_stanford_paper_10s.png")
plt.close()

# 15 s
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
ax = axs
## read data 
data_path = os.path.join(root_data_path, "15s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
#simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
simulations = np.round(np.concatenate([simulation_random, simulation_eager], axis=0), 2)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
reachable_cells_two_step_method = data["reachable_cells_two_step_method"]
print(f"step 15, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}, cells of two step method: {len(reachable_cells_two_step_method)}")

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

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_two_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='yellow', alpha=0.5, label='_nolegend_')

unsafe_polygon = Polygon()
unsafe_patch_1 = [-11, -10, -30, 30]
unsafe_patch_2 = [10, 11, -30, 30]
unsafe_polygon1 = unsafe_polygon.union(Polygon([[unsafe_patch_1[0], unsafe_patch_1[2]], [unsafe_patch_1[0], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[2]]]))
x, y = unsafe_polygon1.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')
unsafe_polygon2 = unsafe_polygon.union(Polygon([[unsafe_patch_2[0], unsafe_patch_2[2]], [unsafe_patch_2[0], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[2]]]))
x, y = unsafe_polygon2.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')

ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')
tikzplotlib.save("single_cell_reachable_set_15s.tex")
plt.savefig("reachable_sets_stanford_paper_15s.png")
plt.close()

"""
# 20 s
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
ax = axs
## read data 
data_path = os.path.join(root_data_path, "20s", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
#simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
simulations = np.round(np.concatenate([simulation_random, simulation_eager], axis=0), 2)
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
"""

# converged
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
ax = axs
## read data 
data_path = os.path.join(root_data_path, "converged", "data.pkl")
data = pickle.load(open(data_path, "rb"))
simulation_p_random = data["simulation_p_random"]
simulation_theta_random = data["simulation_theta_random"]
simulation_p_eager = data["simulation_p_eager"]
simulation_theta_eager = data["simulation_theta_eager"]
simulation_random = np.concatenate([simulation_p_random, simulation_theta_random], axis=1)
simulation_eager = np.concatenate([simulation_p_eager, simulation_theta_eager], axis=1)
#simulations = np.concatenate([simulation_random, simulation_eager], axis=0)
simulations = np.round(np.concatenate([simulation_random, simulation_eager], axis=0), 2)
reachable_cells_baseline = data["reachable_cells_baseline"]
reachable_cells_one_step_method = data["reachable_cells_one_step_method"]
reachable_cells_two_step_method = data["reachable_cells_two_step_method"]
print(f"converged, cells of baseline method: {len(reachable_cells_baseline)}, cells of one step method: {len(reachable_cells_one_step_method)}, cells of two step method: {len(reachable_cells_two_step_method)}")

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

polygon = Polygon()
for idx, cell in enumerate(reachable_cells_two_step_method):
    reachable_patch = [p_lbs[cell[0]], p_lbs[cell[0]]+p_lbs[1]-p_lbs[0], theta_lbs[cell[1]], theta_lbs[cell[1]]+theta_lbs[1]-theta_lbs[0]]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))
x, y = polygon.exterior.xy
ax.fill(x, y, color='yellow', alpha=0.5, label='_nolegend_')

unsafe_polygon = Polygon()
unsafe_patch_1 = [-11, -10, -30, 30]
unsafe_patch_2 = [10, 11, -30, 30]
unsafe_polygon1 = unsafe_polygon.union(Polygon([[unsafe_patch_1[0], unsafe_patch_1[2]], [unsafe_patch_1[0], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[3]], [unsafe_patch_1[1], unsafe_patch_1[2]]]))
x, y = unsafe_polygon1.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')
unsafe_polygon2 = unsafe_polygon.union(Polygon([[unsafe_patch_2[0], unsafe_patch_2[2]], [unsafe_patch_2[0], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[3]], [unsafe_patch_2[1], unsafe_patch_2[2]]]))
x, y = unsafe_polygon2.exterior.xy
ax.fill(x, y, color='pink', alpha=0.5, label='_nolegend_')




ax.set_xlim(-11.0, 11.0)
ax.set_ylim(-30.0, 30.0)
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')
tikzplotlib.save("single_cell_reachable_set_converged.tex")
plt.savefig("reachable_sets_stanford_paper_converged.png")
plt.close()


