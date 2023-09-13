import numpy as np
import os
from datetime import datetime
import logging
from tqdm import tqdm
from collections import defaultdict
import pickle
import arguments


def main():
    # Create the argument parser
    args = arguments.Config
    h = ['system parameters']
    args.add_argument('--p_idx', type=int, default=31, help='Index of p.', hierarchy=h + ['p_idx'])
    args.add_argument('--theta_idx', type=int, default=65, help='Index of theta.', hierarchy=h + ['theta_idx'])
    args.add_argument('--p_range_lb', type=float, default=-11.0, help='Lower bound for p.', hierarchy=h + ['p_lb'])
    args.add_argument('--p_range_ub', type=float, default=+11.0, help='Upper bound for p.', hierarchy=h + ['p_ub'])
    args.add_argument('--p_num_bin', type=int, default=128, help='Number of bins for p.', hierarchy=h + ['p_num_bin'])
    args.add_argument('--theta_range_lb', type=float, default=-30.0, help='Lower bound for theta.', hierarchy=h + ['theta_lb'])
    args.add_argument('--theta_range_ub', type=float, default=+30.0, help='Upper bound for theta.', hierarchy=h + ['theta_ub'])
    args.add_argument('--theta_num_bin', type=int, default=128, help='Number of bins for theta.', hierarchy=h + ['theta_num_bin'])
    args.add_argument('--simulation_samples', type=int, default=10000, help='Number of simulation samples.', hierarchy=h + ['simulation_samples'])
    args.add_argument('--reachability_steps', type=int, default=1, help='Number of reachability steps.', hierarchy=h + ['reachability_steps'])
    args.add_argument('--latent_bounds', type=float, default=0.8, help='Bounds for latent variables.', hierarchy=h + ['latent_bounds'])
    args.add_argument('--p_coeff', type=float, default=-0.74, help='Coefficient for p.', hierarchy=h + ['p_coeff'])
    args.add_argument('--theta_coeff', type=float, default=-0.44, help='Coefficient for theta.', hierarchy=h + ['theta_coeff'])
    args.parse_config()

    p_idx = args['system parameters']['p_idx']
    theta_idx = args['system parameters']['theta_idx']
    p_range_lb = args['system parameters']['p_lb']
    p_range_ub = args['system parameters']['p_ub']
    p_num_bin = args['system parameters']['p_num_bin']
    theta_range_lb = args['system parameters']['theta_lb']
    theta_range_ub = args['system parameters']['theta_ub']
    theta_num_bin = args['system parameters']['theta_num_bin']
    latent_bounds = args['system parameters']['latent_bounds']
    reachability_steps = args['system parameters']['reachability_steps']
    simulation_samples = args['system parameters']['simulation_samples']
    p_coeff = args['system parameters']['p_coeff']
    theta_coeff = args['system parameters']['theta_coeff']
    

    result_path = f"./results/reachable_sets_cell/p_coeff_{p_coeff}_theta_coeff_{theta_coeff}_p_num_bin_{p_num_bin}_theta_num_bin_{theta_num_bin}_steps_{reachability_steps}"
    os.makedirs(result_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(result_path, f"log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a'
    )
    from src.utils_ab_crown import MultiStepVerifier


    p_bins = np.linspace(p_range_lb, p_range_ub, p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    p_ubs = np.array(p_bins[1:], dtype=np.float32)

    theta_bins = np.linspace(theta_range_lb, theta_range_ub, theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
    theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

    verifier = MultiStepVerifier(p_lbs, p_ubs, theta_lbs, theta_ubs, reachability_steps, latent_bounds, simulation_samples, None, p_coeff, theta_coeff)
    logging.info(f"Computing reachable set for cell ({p_idx}, {theta_idx})")
    results = verifier.compute_next_reachable_cells(p_idx, theta_idx)
    reachable_cells = results["reachable_cells"]
    time = results["time"]["whole_time"]
    num_calls_alpha_beta_crown = results["num_calls_alpha_beta_crown"]
    logging.info(f"    Taking {time} seconds and calling alpha-beta-crown {num_calls_alpha_beta_crown} times")
    logging.info(f"    For each call, the alpha-beta-crown setting is: {results['setting_idx_for_each_call']}")
    logging.info(f"    Reachable cells: {reachable_cells}")
    if results['error_during_verification']:
        logging.info(f"    Error during verification, which means the reachable set might not be tightest.")

if __name__ == '__main__':
    main()