import numpy as np
import pickle
import os
from collections import defaultdict
from src.utils import *


p_range_lb = -11.0
p_range_ub = 11.0
p_num_bin = 64
theta_range_lb = -30.0
theta_range_ub = 30.0
theta_num_bin = 64


p_bins = np.linspace(p_range_lb, p_range_ub, p_num_bin+1, endpoint=True)
p_lbs = np.array(p_bins[:-1],dtype=np.float32)
p_ubs = np.array(p_bins[1:], dtype=np.float32)

theta_bins = np.linspace(theta_range_lb, theta_range_ub, theta_num_bin+1, endpoint=True)
theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

# baseline method
baseline_results_path = f"./results/reachable_sets_graph/p_coeff_-0.74_theta_coeff_-0.44_p_num_bin_{p_num_bin}_theta_num_bin_{theta_num_bin}_baseline"
reachable_sets_pickle_file = os.path.join(baseline_results_path, "reachable_sets.pkl")
baseline_verifier = BaselineVerifier(p_lbs, p_ubs, theta_lbs, theta_ubs, csv_file=os.path.join(baseline_results_path, "control_bounds.csv"))
reachable_sets_baseline = defaultdict(set)

for p_idx in range(p_num_bin):
    for theta_idx in range(theta_num_bin):
        reachable_cells = baseline_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True)['reachable_cells']
        reachable_sets_baseline[(p_idx, theta_idx)] = reachable_cells

with open(reachable_sets_pickle_file, "wb") as f:
    pickle.dump(reachable_sets_baseline, f)

# one step method
one_step_results_path_original = f"./results/reachable_sets_graph/p_coeff_-0.74_theta_coeff_-0.44_p_num_bin_128_theta_num_bin_128_steps_1"
one_step_results_path = f"./results/reachable_sets_graph/p_coeff_-0.74_theta_coeff_-0.44_p_num_bin_64_theta_num_bin_64_steps_1"
reachable_sets_one_step_method = defaultdict(set)
with open(os.path.join(one_step_results_path_original, "reachable_sets.pkl"), "rb") as f:
    reachable_sets_one_step_original = pickle.load(f)
    for cell in reachable_sets_one_step_original:
        reachable_sets_original = reachable_sets_one_step_original[cell]
        new_cell = (cell[0]//2, cell[1]//2)
        if reachable_sets_one_step_method[new_cell] == {(-2, -2)}:
            continue
        
        for reachable_cell in reachable_sets_original:
            if reachable_cell[0] >= 0 and reachable_cell[1]>=0:
                reachable_sets_one_step_method[new_cell].add((reachable_cell[0]//2, reachable_cell[1]//2))
            else:
                if reachable_cell == (-2, -2):
                    reachable_sets_one_step_method[new_cell] = {(-2, -2)}
                    break
                reachable_sets_one_step_method[new_cell].add(reachable_cell)
with open(os.path.join(one_step_results_path, "reachable_sets.pkl"), "wb") as f:
    pickle.dump(reachable_sets_one_step_method, f)

# two step method
two_step_results_path_original = f"./results/reachable_sets_graph/p_coeff_-0.74_theta_coeff_-0.44_p_num_bin_128_theta_num_bin_128_steps_2"
two_step_results_path = f"./results/reachable_sets_graph/p_coeff_-0.74_theta_coeff_-0.44_p_num_bin_64_theta_num_bin_64_steps_2"
reachable_sets_two_step_method = defaultdict(set)
with open(os.path.join(two_step_results_path_original, "reachable_sets.pkl"), "rb") as f:
    reachable_sets_two_step_original = pickle.load(f)
    for cell in reachable_sets_two_step_original:
        reachable_sets_original = reachable_sets_two_step_original[cell]
        new_cell = (cell[0]//2, cell[1]//2)
        if reachable_sets_two_step_method[new_cell] == {(-2, -2)}:
            continue
        
        for reachable_cell in reachable_sets_original:
            if reachable_cell[0] >= 0 and reachable_cell[1]>=0:
                reachable_sets_two_step_method[new_cell].add((reachable_cell[0]//2, reachable_cell[1]//2))
            else:
                if reachable_cell == (-2, -2):
                    reachable_sets_two_step_method[new_cell] = {(-2, -2)}
                    break
                reachable_sets_two_step_method[new_cell].add(reachable_cell)
with open(os.path.join(two_step_results_path, "reachable_sets.pkl"), "wb") as f:
    pickle.dump(reachable_sets_two_step_method, f)