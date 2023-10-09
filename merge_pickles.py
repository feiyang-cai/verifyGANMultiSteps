import argparse
import numpy as np
import os
import pickle
from collections import defaultdict
import csv
from src.utils import *


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add the arguments
    parser.add_argument('--p_range_lb', type=float, default=-11.0, help='Lower bound for p_range')
    parser.add_argument('--p_range_ub', type=float, default=+11.0, help='Upper bound for p_range')
    parser.add_argument('--p_num_bin', type=int, default=128, help='Number of bins for p')
    parser.add_argument('--theta_range_lb', type=float, default=-30.0, help='Lower bound for theta_range')
    parser.add_argument('--theta_range_ub', type=float, default=+30.0, help='Upper bound for theta_range')
    parser.add_argument('--theta_num_bin', type=int, default=128, help='Number of bins for theta')
    parser.add_argument('--coeff_p', type=float, default=-0.74, help='Coefficient for p')
    parser.add_argument('--coeff_theta', type=float, default=-0.44, help='Coefficient for theta')

    args = parser.parse_args()

    p_range_lb = args.p_range_lb
    p_range_ub = args.p_range_ub
    p_num_bin = args.p_num_bin
    theta_range_lb = args.theta_range_lb
    theta_range_ub = args.theta_range_ub
    theta_num_bin = args.theta_num_bin


    p_bins = np.linspace(p_range_lb, p_range_ub, p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    p_ubs = np.array(p_bins[1:], dtype=np.float32)

    theta_bins = np.linspace(theta_range_lb, theta_range_ub, theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
    theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

    # baseline method
    baseline_results_path = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{args.p_num_bin}_theta_num_bin_{args.theta_num_bin}_baseline"
    reachable_sets_pickle_file = os.path.join(baseline_results_path, "reachable_sets.pkl")
    baseline_verifier = BaselineVerifier(p_lbs, p_ubs, theta_lbs, theta_ubs, csv_file=os.path.join(baseline_results_path, "control_bounds.csv"))
    reachable_sets_baseline = defaultdict(set)

    for p_idx in range(p_num_bin):
        for theta_idx in range(theta_num_bin):
            reachable_cells = baseline_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True)['reachable_cells']
            reachable_sets_baseline[(p_idx, theta_idx)] = reachable_cells

    with open(reachable_sets_pickle_file, "wb") as f:
        pickle.dump(reachable_sets_baseline, f)
    
    # two step method
    two_step_method_reslts_path = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{args.p_num_bin}_theta_num_bin_{args.theta_num_bin}_steps_2"
    reachable_sets_two_step_method = defaultdict(set)
    reachable_sets_pickle_file = os.path.join(two_step_method_reslts_path, "reachable_sets.pkl")
    file_defining_need_to_comupute = "global_reachable_sets_one_step.pkl"
    with open(file_defining_need_to_comupute, "rb") as f:
        cells_need_to_verify = pickle.load(f)
    for cell in cells_need_to_verify:
        file_path = os.path.join(two_step_method_reslts_path, f"results_p_idx_{cell[0]}_theta_idx_{cell[1]}.pkl")
        data = pickle.load(open(file_path, "rb"))
        reachable_cells = data['reachable_cells']
        reachable_sets_two_step_method[(cell[0], cell[1])] = reachable_cells
    with open(reachable_sets_pickle_file, "wb") as f:
        pickle.dump(reachable_sets_two_step_method, f)



if __name__ == '__main__':
    main()