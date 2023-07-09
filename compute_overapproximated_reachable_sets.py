import argparse
import numpy as np
from src.utils import *
import os


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add the arguments
    parser.add_argument('--p_lb', type=float, default=-10, help='Lower bound of p')
    parser.add_argument('--p_ub', type=float, default=+10, help='Upper bound of p')
    parser.add_argument('--theta_lb', type=float, default=-10, help='Lower bound of theta')
    parser.add_argument('--theta_ub', type=float, default=+10, help='Upper bound of theta')
    parser.add_argument('--p_range_lb', type=float, default=-11.0, help='Lower bound for p_range')
    parser.add_argument('--p_range_ub', type=float, default=+11.0, help='Upper bound for p_range')
    parser.add_argument('--p_num_bin', type=int, default=128, help='Number of bins for p')
    parser.add_argument('--theta_range_lb', type=float, default=-30.0, help='Lower bound for theta_range')
    parser.add_argument('--theta_range_ub', type=float, default=+30.0, help='Upper bound for theta_range')
    parser.add_argument('--theta_num_bin', type=int, default=128, help='Number of bins for theta')

    # Parse the command-line arguments
    args = parser.parse_args()

    p_lb = args.p_lb
    p_ub = args.p_ub
    theta_lb = args.theta_lb
    theta_ub = args.theta_ub
    
    p_range_lb = args.p_range_lb
    p_range_ub = args.p_range_ub
    p_num_bin = args.p_num_bin
    theta_range_lb = args.theta_range_lb
    theta_range_ub = args.theta_range_ub
    theta_num_bin = args.theta_num_bin

    assert p_lb >= p_range_lb and p_ub <= p_range_ub
    assert theta_lb >= theta_range_lb and theta_ub <= theta_range_ub

    p_bins = np.linspace(p_range_lb, p_range_ub, p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    p_ubs = np.array(p_bins[1:], dtype=np.float32)

    theta_bins = np.linspace(theta_range_lb, theta_range_ub, theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
    theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

    # simulation
    network_file_path = "./models/system_model_1.onnx"
    simulator = Simulator(network_file_path)
    samples = 5000
    p = np.random.uniform(p_lb, p_ub, size=(samples, 1)).astype(np.float32)
    theta = np.random.uniform(theta_lb, theta_ub, size=(samples, 1)).astype(np.float32)
    
    # baseline verifier
    network_file_path = "./models/pre_dynamics.onnx"
    csv_file = "./control_bounds.csv"
    results_dir = "./results/compute_overapproximated_reachable_sets/"
    os.makedirs(results_dir, exist_ok=True)

    baseline_verifier = BaselineVerifier(network_file_path, p_lbs, p_ubs, theta_lbs, theta_ubs, csv_file)


    p_bounds = np.array([p_lb, p_ub], dtype=np.float32)
    theta_bounds = np.array([theta_lb, theta_ub], dtype=np.float32)
    reachable_cells = baseline_verifier.get_overlapping_cells_from_intervals(p_bounds, theta_bounds, return_indices=True)
    
    # 0s
    plot_baseline = Plotter(p_lbs, theta_lbs)
    plot_baseline.add_cells(reachable_cells, color='teal', filled=True)
    plot_baseline.add_simulations(p, theta, color='red')
    image_file_path = os.path.join(results_dir, "0s.png")
    plot_baseline.save_figure(image_file_path, x_range=(-11.0, 11.0), y_range=(-30.0, 30.0))

    step = 1
    while True:
        print(f"Computing reachable cells at step {step}")
        reachable_cells_ = set()
        p, theta = simulator.simulate_next_step(p, theta)
        for cell in reachable_cells:
            p_idx, theta_idx = cell
            reachable_cells_ |= baseline_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True)
        if reachable_cells_ == reachable_cells:
            print(f"Converged at step {step}")
            plot_baseline = Plotter(p_lbs, theta_lbs)
            plot_baseline.add_cells(reachable_cells, color='teal', filled=True)
            plot_baseline.add_simulations(p, theta, color='red')
            image_file_path = os.path.join(results_dir, "converged.png")
            plot_baseline.save_figure(image_file_path, x_range=(-11.0, 11.0), y_range=(-30.0, 30.0))
            break
        reachable_cells = reachable_cells_
        plot_baseline = Plotter(p_lbs, theta_lbs)
        plot_baseline.add_cells(reachable_cells, color='teal', filled=True)
        plot_baseline.add_simulations(p, theta, color='red')
        image_file_path = os.path.join(results_dir, f"{step}s.png")
        plot_baseline.save_figure(image_file_path, x_range=(-11.0, 11.0), y_range=(-30.0, 30.0))
        step += 1



if __name__ == '__main__':
    main()