import pickle
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import numpy as np

p_bins = np.linspace(-11, 11, 129, endpoint=True)
p_lbs = np.array(p_bins[:-1],dtype=np.float32)
p_ubs = np.array(p_bins[1:], dtype=np.float32)

theta_bins = np.linspace(-30, 30, 129, endpoint=True)
theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

# simulations 
simulations = pickle.load(open("./results/compare_reachable_sets_for_cell/results_simulation_-11.0_11.0_128_-30.0_30.0_128_31_65.pkl", "rb"))
p_0_step = [round(i[0], 2) for i in simulations["p"].tolist()]
theta_0_step = [round(i[0], 2) for i in simulations["theta"].tolist()]
p_1_step = [round(i[0], 2) for i in simulations["p_1_step"].tolist()]
theta_1_step = [round(i[0], 2) for i in simulations["theta_1_step"].tolist()]
p_2_step = [round(i[0], 2) for i in simulations["p_2_step"].tolist()]
theta_2_step = [round(i[0], 2) for i in simulations["theta_2_step"].tolist()]

# baseline results
baseline_results = pickle.load(open("./results/compare_reachable_sets_for_cell/results_baseline_-11.0_11.0_128_-30.0_30.0_128_31_65.pkl", "rb"))
reachable_cells_baseline_1_step = baseline_results["reachable_cells_1_step"]
reachable_patches_baseline_1_step = baseline_results["reachable_patches_1_step"]
reachable_cells_baseline_2_step = baseline_results["reachable_cells_2_step"]
reachable_patches_baseline_2_step = baseline_results["reachable_patches_2_step"]



fig, axs = plt.subplots(3, 3, figsize=(12, 12))

## 0 step
ax = axs[0, 0]

## plot grids
for p_lb in p_lbs[29:40]:
    X = [p_lb, p_lb]
    Y = [theta_lbs[0], theta_lbs[-1]+theta_lbs[1]-theta_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs[63:88]:
    Y = [theta_lb, theta_lb]
    X = [p_lbs[0], p_lbs[-1]+p_lbs[1]-p_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

ax.scatter(p_0_step, theta_0_step, s=0.05, color='black')
cell = plt.Rectangle((p_lbs[31], theta_lbs[65]), p_lbs[1]-p_lbs[0], theta_lbs[1]-theta_lbs[0], facecolor='none', edgecolor='cornflowerblue', linewidth=2, zorder=10,  label='_nolegend_')
ax.add_patch(cell)
#ax.set_xlabel(r'$p (m)$')
#ax.set_ylabel(r'$\theta (degrees)$')
ax.set_xlim(-6.0, -4.3)
ax.set_ylim(0.0, 11.0)


## 1 step
ax = axs[0, 1]
## plot grids
for p_lb in p_lbs[29:40]:
    X = [p_lb, p_lb]
    Y = [theta_lbs[0], theta_lbs[-1]+theta_lbs[1]-theta_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs[63:88]:
    Y = [theta_lb, theta_lb]
    X = [p_lbs[0], p_lbs[-1]+p_lbs[1]-p_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for patch in reachable_patches_baseline_1_step:
    x = patch[0]
    y = patch[2]
    width = patch[1] - patch[0]
    height = patch[3] - patch[2]
    rec = plt.Rectangle((x, y), width, height, facecolor='lightpink', alpha=0.5, label='_nolegend_')
    ax.add_patch(rec)

for cell_idx in reachable_cells_baseline_1_step:
    rec = plt.Rectangle((p_lbs[cell_idx[0]], theta_lbs[cell_idx[1]]), p_lbs[1]-p_lbs[0], theta_lbs[1]-theta_lbs[0], facecolor='none', edgecolor='cornflowerblue', linewidth=2, zorder=10,  label='_nolegend_')
    ax.add_patch(rec)

ax.scatter(p_1_step, theta_1_step, s=0.05, color='black')
#ax.set_xlabel(r'$p (m)$')
#ax.set_ylabel(r'$\theta (degrees)$')
ax.set_xlim(-6.0, -4.3)
ax.set_ylim(0.0, 11.0)

## 2 step
ax = axs[0, 2]
## plot grids
for p_lb in p_lbs[29:40]:
    X = [p_lb, p_lb]
    Y = [theta_lbs[0], theta_lbs[-1]+theta_lbs[1]-theta_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs[63:88]:
    Y = [theta_lb, theta_lb]
    X = [p_lbs[0], p_lbs[-1]+p_lbs[1]-p_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

reachable_patch = reachable_patches_baseline_2_step[0]
polygon = Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]])

for i in range(1, len(reachable_patches_baseline_2_step)):
    reachable_patch = reachable_patches_baseline_2_step[i]
    polygon = polygon.union(Polygon([[reachable_patch[0], reachable_patch[2]], [reachable_patch[0], reachable_patch[3]], [reachable_patch[1], reachable_patch[3]], [reachable_patch[1], reachable_patch[2]]]))

x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

#for patch in reachable_patches_baseline_2_step:
#    x = patch[0]
#    y = patch[2]
#    width = patch[1] - patch[0]
#    height = patch[3] - patch[2]
#    rec = plt.Rectangle((x, y), width, height, facecolor='lightpink', alpha=0.5, label='_nolegend_')
#    ax.add_patch(rec)

for cell_idx in reachable_cells_baseline_2_step:
    rec = plt.Rectangle((p_lbs[cell_idx[0]], theta_lbs[cell_idx[1]]), p_lbs[1]-p_lbs[0], theta_lbs[1]-theta_lbs[0], facecolor='none', edgecolor='cornflowerblue', linewidth=2, zorder=10,  label='_nolegend_')
    ax.add_patch(rec)
ax.scatter(p_2_step, theta_2_step, s=0.05, color='black')
#ax.set_xlabel(r'$p (m)$')
#ax.set_ylabel(r'$\theta (degrees)$')
ax.set_xlim(-6.0, -4.3)
ax.set_ylim(0.0, 11.0)

# 1-step method

one_step_results = pickle.load(open("./results/compare_reachable_sets_for_cell/results_one_step_-11.0_11.0_128_-30.0_30.0_128_31_65.pkl", "rb"))

## 0 step
ax = axs[1, 0]
ax.axis('off')

## 1 step
ax = axs[1, 1]
## plot grids
for p_lb in p_lbs[29:40]:
    X = [p_lb, p_lb]
    Y = [theta_lbs[0], theta_lbs[-1]+theta_lbs[1]-theta_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs[63:88]:
    Y = [theta_lb, theta_lb]
    X = [p_lbs[0], p_lbs[-1]+p_lbs[1]-p_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)
reachable_cells_1_step = one_step_results["reachable_cells_1_step"]
reachable_verts_1_step = one_step_results["reachable_verts_1_step"]
polygon = Polygon(reachable_verts_1_step[0])

for i in range(1, len(reachable_verts_1_step)):
    polygon = polygon.union(Polygon(reachable_verts_1_step[i]))

x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

for cell_idx in reachable_cells_1_step:
    rec = plt.Rectangle((p_lbs[cell_idx[0]], theta_lbs[cell_idx[1]]), p_lbs[1]-p_lbs[0], theta_lbs[1]-theta_lbs[0], facecolor='none', edgecolor='cornflowerblue', linewidth=2, zorder=10,  label='_nolegend_')
    ax.add_patch(rec)
ax.scatter(p_1_step, theta_1_step, s=0.05, color='black')
#ax.set_xlabel(r'$p (m)$')
#ax.set_ylabel(r'$\theta (degrees)$')
ax.set_xlim(-6.0, -4.3)
ax.set_ylim(0.0, 11.0)

## 2 step
ax = axs[1, 2]
## plot grids
for p_lb in p_lbs[29:40]:
    X = [p_lb, p_lb]
    Y = [theta_lbs[0], theta_lbs[-1]+theta_lbs[1]-theta_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs[63:88]:
    Y = [theta_lb, theta_lb]
    X = [p_lbs[0], p_lbs[-1]+p_lbs[1]-p_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)
reachable_cells_2_step = one_step_results["reachable_cells_2_step"]
reachable_verts_2_step = one_step_results["reachable_verts_2_step"]
polygon = Polygon(reachable_verts_2_step[0])

for i in range(1, len(reachable_verts_2_step)):
    polygon = polygon.union(Polygon(reachable_verts_2_step[i]))

x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

for cell_idx in reachable_cells_2_step:
    rec = plt.Rectangle((p_lbs[cell_idx[0]], theta_lbs[cell_idx[1]]), p_lbs[1]-p_lbs[0], theta_lbs[1]-theta_lbs[0], facecolor='none', edgecolor='cornflowerblue', linewidth=2, zorder=10,  label='_nolegend_')
    ax.add_patch(rec)
ax.scatter(p_2_step, theta_2_step, s=0.05, color='black')
#ax.set_xlabel(r'$p (m)$')
#ax.set_ylabel(r'$\theta (degrees)$')
ax.set_xlim(-6.0, -4.3)
ax.set_ylim(0.0, 11.0)


# 2-step method

two_step_results = pickle.load(open("./results/compare_reachable_sets_for_cell/results_two_step_-11.0_11.0_128_-30.0_30.0_128_31_65.pkl", "rb"))

## 0 step
ax = axs[2, 0]
ax.axis('off')

## 1 step
ax = axs[2, 1]
ax.axis('off')

## 2 step
ax = axs[2, 2]
## plot grids
for p_lb in p_lbs[29:40]:
    X = [p_lb, p_lb]
    Y = [theta_lbs[0], theta_lbs[-1]+theta_lbs[1]-theta_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs[63:88]:
    Y = [theta_lb, theta_lb]
    X = [p_lbs[0], p_lbs[-1]+p_lbs[1]-p_lbs[0]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)
reachable_cells_2_step = two_step_results["reachable_cells_2_step"]
reachable_verts_2_step = two_step_results["reachable_verts_2_step"]
polygon = Polygon(reachable_verts_2_step[0])

for i in range(1, len(reachable_verts_2_step)):
    polygon = polygon.union(Polygon(reachable_verts_2_step[i]))

x, y = polygon.exterior.xy
ax.fill(x, y, color='lightpink', alpha=0.5, label='_nolegend_')

for cell_idx in reachable_cells_2_step:
    rec = plt.Rectangle((p_lbs[cell_idx[0]], theta_lbs[cell_idx[1]]), p_lbs[1]-p_lbs[0], theta_lbs[1]-theta_lbs[0], facecolor='none', edgecolor='cornflowerblue', linewidth=2, zorder=10,  label='_nolegend_')
    ax.add_patch(rec)
ax.scatter(p_2_step, theta_2_step, s=0.05, color='black')
ax.set_xlabel(r'$p (m)$')
ax.set_ylabel(r'$\theta (degrees)$')
ax.set_xlim(-6.0, -4.3)
ax.set_ylim(0.0, 11.0)


plt.tight_layout()
import tikzplotlib
tikzplotlib.save("single_cell_reachable_set.tex")

plt.savefig("demo.png")

