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
    parser.add_argument('--coeff_p', type=float, default=-0.2, help='Coefficient for p')
    parser.add_argument('--coeff_theta', type=float, default=-0.1, help='Coefficient for theta')
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

    result_path = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_steps_{args.reachability_steps}_server_{server_id}_{server_total_num}"
    os.makedirs(result_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(result_path, f"log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a'
    )

    emergency_save_file = os.path.join(result_path,
                                       f"reachable_set_{int(p_lbs[0])}_{int(p_ubs[-1])}_{len(p_lbs)}_{int(theta_lbs[0])}_{int(theta_ubs[-1])}_{len(theta_lbs)}_emergency_save.pkl")
    whole_graph_file = os.path.join(result_path,
                                    f"reachable_set_{int(p_lbs[0])}_{int(p_ubs[-1])}_{len(p_lbs)}_{int(theta_lbs[0])}_{int(theta_ubs[-1])}_{len(theta_lbs)}.pkl")


    verifier = MultiStepVerifier(network_path, p_lbs, p_ubs, theta_lbs, theta_ubs)

    reachable_set = defaultdict(set)

    if os.path.exists(whole_graph_file):
        logging.info("Reachable set already exists, exiting")
        sys.exit(0)

    if os.path.exists(emergency_save_file):
        logging.info("Emergency save file exists, loading from it")
        with open(emergency_save_file, 'rb') as f:
            reachable_set = pickle.load(f)

    start_point = len(p_lbs) // server_total_num * (server_id-1)
    end_point = len(p_lbs) // server_total_num * (server_id)

    try:
        for p_idx in tqdm(range(start_point, end_point)):
            line_file = os.path.join(result_path,
                                    f"reachable_set_{int(p_lbs[0])}_{int(p_ubs[-1])}_{len(p_lbs)}_{int(theta_lbs[0])}_{int(theta_ubs[-1])}_{len(theta_lbs)}_line_{p_idx}.pkl")
            
            if os.path.exists(line_file):
                continue

            for theta_idx in (pbar := tqdm(range(len(theta_lbs)), leave=False)):

                # Skip if already computed
                if (p_idx, theta_idx) in reachable_set:
                    continue

                logging.info(f"Computing reachable set for cell ({p_idx}, {theta_idx})")
                results = verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True, pbar=pbar, return_tolerance=True)
                if results['split_tolerance'] is None:
                    logging.error(f"Error in computing reachable set for cell ({p_idx}, {theta_idx}) trying all split tolerances")
                else:
                    logging.info(f"Computed reachable set for cell ({p_idx}, {theta_idx}) using split tolerance {results['split_tolerance']}, and the reachable set has {len(results['reachable_cells'])} cells")
                reachable_set[(p_idx, theta_idx)] = results['reachable_cells']
            
            with open(line_file, 'wb') as f:
                pickle.dump(reachable_set, f)

    except:
        logging.error(f"Error in computing reachable set for cell ({p_idx}, {theta_idx}), saving the reachable set to file")
        with open(emergency_save_file, 'wb') as f:
            pickle.dump(reachable_set, f)
        logging.error("Emergency save done, please restart the program")
        raise Exception("Emergency save done, please restart the program")


if __name__ == '__main__':
    main()