import argparse
import numpy as np
from src.utils import *
import os
import pickle


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

    # Parse the command-line arguments
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

    results_dir = f"./results/unsafe_initial_cells/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{p_num_bin}_theta_num_bin_{theta_num_bin}"
    os.makedirs(results_dir, exist_ok=True)

    # baseline method
    baseline_csv_file = f"./results/baseline_control_bounds/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{p_num_bin}_theta_num_bin_{theta_num_bin}.csv"
    assert os.path.exists(baseline_csv_file), f"Baseline csv file {baseline_csv_file} does not exist"
    baseline_verifier = BaselineVerifier(p_lbs, p_ubs, theta_lbs, theta_ubs, csv_file=baseline_csv_file)

    reachable_sets = defaultdict(set)

    for p_idx in range(len(p_lbs)):
        for theta_idx in range(len(theta_lbs)):
            results = baseline_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True)
            if results['out_of_p_safety_bounds']:
                reachable_sets[(p_idx, theta_idx)].add((-1, -1))
                continue
                
            reachable_sets[(p_idx, theta_idx)] |= results['reachable_cells']

    isSafe = compute_unsafe_cells(reachable_sets, p_lbs, p_ubs, theta_lbs, theta_ubs)
    unsafe_cells = []
    for p_idx in range(len(p_lbs)):
        for theta_idx in range(len(theta_lbs)):
            if isSafe[p_idx, theta_idx] == 0:
                unsafe_cells.append((p_idx, theta_idx))

    print(f"Number of unsafe cells for baseline method: {len(unsafe_cells)}") 
    plotter = Plotter(p_lbs, theta_lbs)
    plotter_together = Plotter(p_lbs, theta_lbs)
    plotter.add_cells(unsafe_cells, color='red', filled=True)
    plotter_together.add_cells(unsafe_cells, color='yellow', filled=True)
    file_path = os.path.join(results_dir, f"unsafe_cells_baseline.png")
    plotter.save_figure(file_path, x_range=(-11, 11), y_range=(-30, 30))

    # one-step method
    reachable_sets = defaultdict(set)
    step_1_results_path = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{args.p_num_bin}_theta_num_bin_{args.theta_num_bin}_steps_1"
    reachable_sets_pickle_file = os.path.join(step_1_results_path, "reachable_sets.pkl")
    one_step_verifier = MultiStepVerifier(p_lbs, p_ubs, theta_lbs, theta_ubs, reachable_cells_path=reachable_sets_pickle_file)
    for p_idx in range(len(p_lbs)):
        for theta_idx in range(len(theta_lbs)):
            results = one_step_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True)
            if p_idx == 3 and theta_idx == 63:
                print(results)
            if results['reachable_cells'] == set():
                print("    No reachable cells (error occurs), using baseline method to over-approximate")
                results = baseline_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True)

            if results['out_of_p_safety_bounds']:
                reachable_sets[(p_idx, theta_idx)].add((-1, -1))
                continue
                
            reachable_sets[(p_idx, theta_idx)] |= results['reachable_cells']

    isSafe = compute_unsafe_cells(reachable_sets, p_lbs, p_ubs, theta_lbs, theta_ubs)
    unsafe_cells = []
    for p_idx in range(len(p_lbs)):
        for theta_idx in range(len(theta_lbs)):
            if isSafe[p_idx, theta_idx] == 0:
                unsafe_cells.append((p_idx, theta_idx))
    print(f"Number of unsafe cells for one-step method: {len(unsafe_cells)}") 
    plotter = Plotter(p_lbs, theta_lbs)
    plotter.add_cells(unsafe_cells, color='red', filled=True)
    plotter_together.add_cells(unsafe_cells, color='red', filled=True)
    file_path = os.path.join(results_dir, f"unsafe_cells_one_step.png")
    plotter.save_figure(file_path, x_range=(-11, 11), y_range=(-30, 30))
    file_path = os.path.join(results_dir, f"unsafe_cells.png")
    plotter_together.add_patches([[-9, 9, -9, 9]], color='blue')
    plotter_together.save_figure(file_path, x_range=(-11, 11), y_range=(-30, 30))

if __name__ == '__main__':
    main()
