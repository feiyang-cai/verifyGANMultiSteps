import argparse
import numpy as np
from src.utils import *
import onnxruntime as ort
import os
import pickle

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add the arguments
    parser.add_argument('--p_idx', type=int, default=30, help='Cell index of p')
    parser.add_argument('--theta_idx', type=int, default=65, help='Cell index of theta')
    parser.add_argument('--p_range_lb', type=float, default=-11.0, help='Lower bound for p_range')
    parser.add_argument('--p_range_ub', type=float, default=+11.0, help='Upper bound for p_range')
    parser.add_argument('--p_num_bin', type=int, default=128, help='Number of bins for p')
    parser.add_argument('--theta_range_lb', type=float, default=-30.0, help='Lower bound for theta_range')
    parser.add_argument('--theta_range_ub', type=float, default=+30.0, help='Upper bound for theta_range')
    parser.add_argument('--theta_num_bin', type=int, default=128, help='Number of bins for theta')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    p_idx = args.p_idx
    theta_idx = args.theta_idx
    p_range_lb = args.p_range_lb
    p_range_ub = args.p_range_ub
    p_num_bin = args.p_num_bin
    theta_range_lb = args.theta_range_lb
    theta_range_ub = args.theta_range_ub
    theta_num_bin = args.theta_num_bin

    assert p_idx >= 0 and p_idx < p_num_bin
    assert theta_idx >= 0 and theta_idx < theta_num_bin

    p_bins = np.linspace(p_range_lb, p_range_ub, p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    p_ubs = np.array(p_bins[1:], dtype=np.float32)

    theta_bins = np.linspace(theta_range_lb, theta_range_ub, theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
    theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

    results_dir = "./results/compare_reachable_sets_for_cell/"
    os.makedirs(results_dir, exist_ok=True)
    print(p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx])

    samples = 1000
    p = np.random.uniform(p_lbs[p_idx], p_ubs[p_idx], size=(samples, 1)).astype(np.float32)
    theta = np.random.uniform(theta_lbs[theta_idx], theta_ubs[theta_idx], size=(samples, 1)).astype(np.float32)

    # initialize the reachable set
    plot_baseline = Plotter(p_lbs, theta_lbs)
    plot_one_step_method = Plotter(p_lbs, theta_lbs)
    plot_two_step_method = Plotter(p_lbs, theta_lbs)
    plot_combined = Plotter(p_lbs, theta_lbs)

    #plot_baseline.add_patches([(p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx])], color='gray')
    plot_baseline.add_simulations(p, theta, color='red', label='Simulations')
    plot_baseline.add_cells([(p_idx, theta_idx)], color='gray', label=r'Overlapping cells (0s)')

    #plot_one_step_method.add_patches([(p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx])], color='blue')
    plot_one_step_method.add_simulations(p, theta, color='red', label='Simulations')
    plot_one_step_method.add_cells([(p_idx, theta_idx)], color='gray', label=r'Overlapping cells (0s)')

    
    #plot_two_step_method.add_patches([(p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx])], color='yellow')
    plot_two_step_method.add_simulations(p, theta, color='red', label='Simulations')
    plot_two_step_method.add_cells([(p_idx, theta_idx)], color='gray', label=r'Overlapping cells (0s)')

    
    #plot_combined.add_patches([(p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx])], color='gray')
    plot_combined.add_cells([(p_idx, theta_idx)], color='gray')
    plot_combined.add_simulations(p, theta, color='red', label='Simulations')
    figure_combined_file_path = results_dir + f"plot_combined_{p_range_lb}_{p_range_ub}_{p_num_bin}_{theta_range_lb}_{theta_range_ub}_{theta_num_bin}_{p_idx}_{theta_idx}.png"

    # simulation
    print("Simulating the system...")
    network_file_path = "./models/system_model_1.onnx"

    simulator = Simulator(network_file_path)
    # first step
    p_1_step, theta_1_step = simulator.simulate_next_step(p, theta)
    # second step
    p_2_step, theta_2_step = simulator.simulate_next_step(p_1_step, theta_1_step)
    
    
    # baseline verifier
    results_baseline_file_path = results_dir + f"results_baseline_{p_range_lb}_{p_range_ub}_{p_num_bin}_{theta_range_lb}_{theta_range_ub}_{theta_num_bin}_{p_idx}_{theta_idx}.pkl"
    figure_baseline_file_path = results_dir + f"figure_baseline_{p_range_lb}_{p_range_ub}_{p_num_bin}_{theta_range_lb}_{theta_range_ub}_{theta_num_bin}_{p_idx}_{theta_idx}.png"
    try:
        with open(results_baseline_file_path, 'rb') as f:
            results_baseline = pickle.load(f)
            reachable_cells_baseline_1_step = results_baseline['reachable_cells_1_step']
            reachable_cells_baseline_2_step = results_baseline['reachable_cells_2_step']
            reachable_patches_baseline_1_step = results_baseline['reachable_patches_1_step']
            reachable_patches_baseline_2_step = results_baseline['reachable_patches_2_step']

    except:
        network_file_path = "./models/pre_dynamics.onnx"
        baseline_verifier = BaselineVerifier(network_file_path, p_lbs, p_ubs, theta_lbs, theta_ubs)
        # first step
        print("Computing reachable set for the 1 step using baseline method...")
        reachable_cells_baseline_1_step,  p_bounds_baseline_1_step, theta_bounds_baseline_1_step = \
            baseline_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True, return_bounds=True)
        reachable_patches_baseline_1_step = [(p_bounds_baseline_1_step[0], p_bounds_baseline_1_step[1], theta_bounds_baseline_1_step[0], theta_bounds_baseline_1_step[1])]
    
        # second step
        print("Computing reachable set for the 2 step using baseline method...")
        reachable_cells_baseline_2_step = set()
        reachable_patches_baseline_2_step = []


        for idx, cell in enumerate(reachable_cells_baseline_1_step):
            print("   Computing reachable set for the 2 step using baseline method for cell {} out of {}".format(idx+1, len(reachable_cells_baseline_1_step)))
            p_idx_1, theta_idx_1 = cell
            reachable_cells_baseline_2_step_single,  p_bounds_baseline_2_step_single, theta_bounds_baseline_2_step_single = \
                baseline_verifier.compute_next_reachable_cells(p_idx_1, theta_idx_1, return_indices=True, return_bounds=True)
            reachable_cells_baseline_2_step = reachable_cells_baseline_2_step.union(reachable_cells_baseline_2_step_single)
            reachable_patches_baseline_2_step.append((p_bounds_baseline_2_step_single[0], p_bounds_baseline_2_step_single[1], theta_bounds_baseline_2_step_single[0], theta_bounds_baseline_2_step_single[1]))

        print("Saving the results...")
        results_baseline = dict()
        results_baseline['reachable_cells_1_step'] = reachable_cells_baseline_1_step
        results_baseline['reachable_cells_2_step'] = reachable_cells_baseline_2_step
        results_baseline['reachable_patches_1_step'] = reachable_patches_baseline_1_step
        results_baseline['reachable_patches_2_step'] = reachable_patches_baseline_2_step
        with open(results_baseline_file_path, 'wb') as f:
            pickle.dump(results_baseline, f)
        

    plot_baseline.add_patches(reachable_patches_baseline_1_step, color='gray')
    plot_baseline.add_cells(reachable_cells_baseline_1_step, color='teal', label=r'Overlapping cells (1s)')
    plot_baseline.add_simulations(p_1_step, theta_1_step, color='red')

    plot_baseline.add_patches(reachable_patches_baseline_2_step, color='gray')
    plot_baseline.add_cells(reachable_cells_baseline_2_step, color='red', label=r'Overlapping cells (2s)')
    plot_baseline.add_simulations(p_2_step, theta_2_step, color='red')
    plot_baseline.save_figure(figure_baseline_file_path)

    plot_combined.add_patches(reachable_patches_baseline_1_step, color='gray', label=r'Overapproximated state (baseline method)')
    plot_combined.add_cells(reachable_cells_baseline_1_step, color='green', label=r'Overlapping cells (baseline method)')

    plot_combined.add_patches(reachable_patches_baseline_2_step, color='gray')
    plot_combined.add_cells(reachable_cells_baseline_2_step, color='green')

    # 1 step verifier
    results_one_step_method_file_path = results_dir + f"results_one_step_{p_range_lb}_{p_range_ub}_{p_num_bin}_{theta_range_lb}_{theta_range_ub}_{theta_num_bin}_{p_idx}_{theta_idx}.pkl"
    figure_one_step_method_file_path = results_dir + f"figure_one_step_{p_range_lb}_{p_range_ub}_{p_num_bin}_{theta_range_lb}_{theta_range_ub}_{theta_num_bin}_{p_idx}_{theta_idx}.png"

    try:
        with open(results_one_step_method_file_path, 'rb') as f:
            results_one_step_method = pickle.load(f)
            reachable_cells_one_step_method_1_step = results_one_step_method['reachable_cells_1_step']
            reachable_cells_one_step_method_2_step = results_one_step_method['reachable_cells_2_step']
            reachable_verts_one_step_method_1_step = results_one_step_method['reachable_verts_1_step']
            reachable_verts_one_step_method_2_step = results_one_step_method['reachable_verts_2_step']

    except:
        network_file_path = "./models/system_model_1.onnx"
        one_step_verifier = MultiStepVerifier(network_file_path, p_lbs, p_ubs, theta_lbs, theta_ubs)

        # first step
        print("Computing reachable set for the 1 step using one step method...")
        reachable_cells_one_step_method_1_step, reachable_verts_one_step_method_1_step = one_step_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True, return_verts=True)

        # second step
        print("Computing reachable set for the 2 step using one step method...")
        reachable_cells_one_step_method_2_step = set()
        reachable_verts_one_step_method_2_step = []

        for idx, cell in enumerate(reachable_cells_one_step_method_1_step):
            print("   Computing reachable set for the 2 step using one step method for cell {} out of {}".format(idx+1, len(reachable_cells_one_step_method_1_step)))
            p_idx_1, theta_idx_1 = cell
            reachable_cells_one_step_method_2_step_single, reachable_verts_one_step_method_2_step_single = \
                one_step_verifier.compute_next_reachable_cells(p_idx_1, theta_idx_1, return_indices=True, return_verts=True)
            reachable_cells_one_step_method_2_step = reachable_cells_one_step_method_2_step.union(reachable_cells_one_step_method_2_step_single)
            reachable_verts_one_step_method_2_step.extend(reachable_verts_one_step_method_2_step_single)

        print("Saving the results...")
        results_one_step_method = dict()
        results_one_step_method['reachable_cells_1_step'] = reachable_cells_one_step_method_1_step
        results_one_step_method['reachable_cells_2_step'] = reachable_cells_one_step_method_2_step
        results_one_step_method['reachable_verts_1_step'] = reachable_verts_one_step_method_1_step
        results_one_step_method['reachable_verts_2_step'] = reachable_verts_one_step_method_2_step
        with open(results_one_step_method_file_path, 'wb') as f:
            pickle.dump(results_one_step_method, f)

    plot_one_step_method.add_verts(reachable_verts_one_step_method_1_step, color='gray')
    plot_one_step_method.add_cells(reachable_cells_one_step_method_1_step, color='teal', label=r'Overlapping cells (1s)')
    plot_one_step_method.add_simulations(p_1_step, theta_1_step, color='red')
    plot_one_step_method.add_verts(reachable_verts_one_step_method_2_step, color='gray')
    plot_one_step_method.add_cells(reachable_cells_one_step_method_2_step, color='red', label=r'Overlapping cells (2s)')
    plot_one_step_method.add_simulations(p_2_step, theta_2_step, color='red')

    plot_one_step_method.save_figure(figure_one_step_method_file_path)

    plot_combined.add_verts(reachable_verts_one_step_method_1_step, color='blue', label=r'Overapproximated state (1-step method)')
    plot_combined.add_cells(reachable_cells_one_step_method_1_step, color='blue', label=r'Overlapping cells (1-step method)')
    plot_combined.add_verts(reachable_verts_one_step_method_2_step, color='blue')
    plot_combined.add_cells(reachable_cells_one_step_method_2_step, color='blue')


    # 2 step verifier
    results_two_step_method_file_path = results_dir + f"results_two_step_{p_range_lb}_{p_range_ub}_{p_num_bin}_{theta_range_lb}_{theta_range_ub}_{theta_num_bin}_{p_idx}_{theta_idx}.pkl"
    figure_two_step_method_file_path = results_dir + f"figure_two_step_{p_range_lb}_{p_range_ub}_{p_num_bin}_{theta_range_lb}_{theta_range_ub}_{theta_num_bin}_{p_idx}_{theta_idx}.png"
    try:
        with open(results_two_step_method_file_path, 'rb') as f:
            results_two_step_method = pickle.load(f)
            #reachable_cells_two_step_method_1_step = results_two_step_method['reachable_cells_1_step']
            reachable_cells_two_step_method_2_step = results_two_step_method['reachable_cells_2_step']
            #reachable_verts_two_step_method_1_step = results_two_step_method['reachable_verts_1_step']
            reachable_verts_two_step_method_2_step = results_two_step_method['reachable_verts_2_step']

    except:
        network_file_path = "./models/system_model_2.onnx"
        two_step_verifier = MultiStepVerifier(network_file_path, p_lbs, p_ubs, theta_lbs, theta_ubs)

        # second step
        print("Computing reachable set for the 2 step using two step method...")
        reachable_cells_two_step_method_2_step, reachable_verts_two_step_method_2_step = two_step_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True, return_verts=True)

        print("Saving the results...")
        results_two_step_method = dict()
        #results_two_step_method['reachable_cells_1_step'] = reachable_cells_one_step_method_1_step
        results_two_step_method['reachable_cells_2_step'] = reachable_cells_two_step_method_2_step
        #results_two_step_method['reachable_verts_1_step'] = reachable_verts_one_step_method_1_step
        results_two_step_method['reachable_verts_2_step'] = reachable_verts_two_step_method_2_step
        with open(results_two_step_method_file_path, 'wb') as f:
            pickle.dump(results_two_step_method, f)

    plot_two_step_method.add_verts(reachable_verts_one_step_method_1_step, color='gray')
    plot_two_step_method.add_cells(reachable_cells_one_step_method_1_step, color='teal', label=r'Overlapping cells (1s)')
    plot_two_step_method.add_simulations(p_1_step, theta_1_step, color='red')

    plot_two_step_method.add_verts(reachable_verts_two_step_method_2_step, color='gray')
    plot_two_step_method.add_cells(reachable_cells_two_step_method_2_step, color='red', label=r'Overlapping cells (2s)')
    plot_two_step_method.add_simulations(p_2_step, theta_2_step, color='red')
    plot_two_step_method.save_figure(figure_two_step_method_file_path)

    plot_combined.add_verts(reachable_verts_two_step_method_2_step, color='yellow', label=r'Overapproximated state (2-step method)')
    plot_combined.add_cells(reachable_cells_two_step_method_2_step, color='yellow', label=r'Overlapping cells (2-step method)')
    plot_combined.add_simulations(p_1_step, theta_1_step, color='red')
    plot_combined.add_simulations(p_2_step, theta_2_step, color='red')
    plot_combined.save_figure(figure_combined_file_path)
    
if __name__ == '__main__':
    main()