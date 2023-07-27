import numpy as np
import argparse
import os
from src.utils import *
from collections import defaultdict
import pickle
import sys
import logging
from datetime import datetime
import multiprocessing
import queue
import threading

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

    # settings / optimizations
    num_cores = multiprocessing.cpu_count()

    try:
        num_cores = len(os.sched_getaffinity(0)) # doesn't work on some unix platforms
    except AttributeError:
        pass
    
    num_threads = num_cores - 1 # leave one core for the main process

    # set network path
    network_path = f"./models/system_model_{args.reachability_steps}_{args.coeff_p}_{args.coeff_theta}.onnx"
    assert os.path.exists(network_path)

    result_path = f"./results/reachable_sets_for_cell/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_steps_{args.reachability_steps}_server_{server_id}_{server_total_num}"
    os.makedirs(result_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # set logging
    log_filename = os.path.join(result_path, f"log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a'
    )

    def worker():
        # initialize verifier
        logging.info(f"Initializing verifier for {threading.current_thread().name}...")
        verifier = MultiStepVerifier(network_path, p_lbs, p_ubs, theta_lbs, theta_ubs)

        while True:
            try:
                p_idx, theta_idx = cell_queue.get(block=False)
            except queue.Empty:
                # no more cells to process
                logging.info(f"Thread {threading.current_thread().name} finished.")
                break
            
            cell_results_path = os.path.join(result_path, f"cell_{p_idx}_{theta_idx}.pkl")
            if os.path.exists(cell_results_path):
                # cell already computed
                logging.info(f"Cell ({p_idx}, {theta_idx}) already computed in {threading.current_thread().name}.")
                continue

            # compute reachable set for the cell
            logging.info(f"Computing reachable set for cell ({p_idx}, {theta_idx}) in {threading.current_thread().name}...")
            results = verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True, return_tolerance=True, single_thread=True)

            # save results
            with open(cell_results_path, "wb") as f:
                pickle.dump(results, f) 
            logging.info(f"Cell ({p_idx}, {theta_idx}) computed in {threading.current_thread().name}, taking {results['time_dict']['total_time']} seconds.")

    logging.info(f"Starting {num_threads} threads in server {server_id}/{server_total_num}...")

    cell_queue = queue.Queue()

    # enqueue all cells
    # the computed cells will not be computed
    # TODO: have a cloud drive to store the results and all the servers can access it
    for p_idx in range(len(p_lbs)):
        for theta_idx in range(len(theta_lbs)):
            cell_queue.put((p_idx, theta_idx))
    
    # Create and start worker threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All tasks have been completed.")

if __name__ == "__main__":
    main()
