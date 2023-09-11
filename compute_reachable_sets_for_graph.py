import numpy as np
import argparse
import os
from src.utils import *
from tqdm import tqdm
from collections import defaultdict
import pickle
import sys
import logging
from datetime import datetime


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
    parser.add_argument('--reachability_steps', type=int, default=1, help='Number of reachability steps')
    parser.add_argument('--coeff_p', type=float, default=-0.74, help='Coefficient for p')
    parser.add_argument('--coeff_theta', type=float, default=-0.44, help='Coefficient for theta')
    parser.add_argument('--server_id', type=int, default=1, help='Server ID')
    parser.add_argument('--server_total_num', type=int, default=1, help='Total number of servers')
    
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

    assert len(p_lbs) % args.server_total_num == 0

    server_id = args.server_id
    server_total_num = args.server_total_num
    assert server_id >= 1 and server_id <= server_total_num

    network_path = f"./models/system_model_{args.reachability_steps}_{args.coeff_p}_{args.coeff_theta}.onnx"
    print(f"Loading network from {network_path}")
    assert os.path.exists(network_path)

    result_path = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{args.p_num_bin}_theta_num_bin_{args.theta_num_bin}_steps_{args.reachability_steps}"
    os.makedirs(result_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(result_path, f"log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a'
    )

    verifier = MultiStepVerifier(p_lbs, p_ubs, theta_lbs, theta_ubs, network_path)

    reachable_set = defaultdict(set)

    start_point = len(p_lbs) // server_total_num * (server_id-1)
    end_point = len(p_lbs) // server_total_num * (server_id)

    for p_idx in tqdm(range(start_point, end_point)):
        for theta_idx in (pbar := tqdm(range(len(theta_lbs)), leave=False)):

            # file_path
            file = os.path.join(result_path, f"results_p_idx_{p_idx}_theta_idx_{theta_idx}.pkl")

            # Skip if already computed
            if os.path.exists(file):
                logging.info(f"Computing rechable set for cell ({p_idx}, {theta_idx})...")
                continue

            logging.info(f"Computing reachable set for cell ({p_idx}, {theta_idx})")
            start_tol = 1e-4 if args.reachability_steps > 1 else 1e-7
            results = verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True, pbar=pbar, return_tolerance=True, start_tol=start_tol)

            if results['split_tolerance'] == -1.0:
                logging.error(f"Error in computing reachable set for cell ({p_idx}, {theta_idx}) trying all split tolerances")
            else:
                logging.info(f"Computed reachable set for cell ({p_idx}, {theta_idx}) using split tolerance {results['split_tolerance']}, and the reachable set has {len(results['reachable_cells'])} cells")

            with open(file, "wb") as f:
                pickle.dump(results, f)
            

if __name__ == '__main__':
    main()