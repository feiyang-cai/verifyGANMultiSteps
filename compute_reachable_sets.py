import argparse
import numpy as np
from src.utils import *
import os
import pickle


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add the arguments
    parser.add_argument('--p_lb', type=float, default=-10.0, help='Lower bound of p')
    parser.add_argument('--p_ub', type=float, default=+10.0, help='Upper bound of p')
    parser.add_argument('--theta_lb', type=float, default=-10.0, help='Lower bound of theta')
    parser.add_argument('--theta_ub', type=float, default=+10.0, help='Upper bound of theta')
    parser.add_argument('--p_range_lb', type=float, default=-11.0, help='Lower bound for p_range')
    parser.add_argument('--p_range_ub', type=float, default=+11.0, help='Upper bound for p_range')
    parser.add_argument('--p_num_bin', type=int, default=128, help='Number of bins for p')
    parser.add_argument('--theta_range_lb', type=float, default=-30.0, help='Lower bound for theta_range')
    parser.add_argument('--theta_range_ub', type=float, default=+30.0, help='Upper bound for theta_range')
    parser.add_argument('--theta_num_bin', type=int, default=128, help='Number of bins for theta')
    parser.add_argument('--coeff_p', type=float, default=-0.74, help='Coefficient for p')
    parser.add_argument('--coeff_theta', type=float, default=-0.44, help='Coefficient for theta')
    parser.add_argument('--add_random_simulations', type=bool, default=True, help='Add random simulations')
    parser.add_argument('--add_eagerly_searching_simulations', type=bool, default=True, help='Add eagerly serching simulations')

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

    results_dir = f"./results/compute_reachable_sets/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_lb_{p_lb}_p_ub_{p_ub}_theta_lb_{theta_lb}_theta_ub_{theta_ub}_p_range_lb_{p_range_lb}_p_range_ub_{p_range_ub}_p_num_bin_{p_num_bin}_theta_range_lb_{theta_range_lb}_theta_range_ub_{theta_range_ub}_theta_num_bin_{theta_num_bin}"
    os.makedirs(results_dir, exist_ok=True)

    global_reachable_sets_baseline = set()
    global_reachable_sets_one_step = set()
    global_reachable_sets_two_step = set()

    # baseline method 
    dir = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{args.p_num_bin}_theta_num_bin_{args.theta_num_bin}_baseline"
    baseline_pickle_file = os.path.join(dir, "reachable_sets.pkl")
    baseline_csv_file = os.path.join(dir, "control_bounds.csv")
    assert os.path.exists(baseline_csv_file), f"Baseline csv file {baseline_csv_file} does not exist"
    baseline_verifier = BaselineVerifier(p_lbs, p_ubs, theta_lbs, theta_ubs, csv_file=baseline_csv_file)

    assert os.path.exists(baseline_pickle_file), 'The result file for baseline method does not exist'
    reachable_sets_baseline = pickle.load(open(baseline_pickle_file, "rb"))

    # one step method
    dir = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{args.p_num_bin}_theta_num_bin_{args.theta_num_bin}_steps_1"
    one_step_pickle_file = os.path.join(dir, "reachable_sets.pkl")

    assert os.path.exists(one_step_pickle_file), 'The result file for one step method does not exist'
    reachable_sets_one_step = pickle.load(open(one_step_pickle_file, "rb"))

    # two step method
    dir = f"./results/reachable_sets_graph/p_coeff_{args.coeff_p}_theta_coeff_{args.coeff_theta}_p_num_bin_{args.p_num_bin}_theta_num_bin_{args.theta_num_bin}_steps_2"
    two_step_pickle_file = os.path.join(dir, "reachable_sets.pkl")

    assert os.path.exists(two_step_pickle_file), 'The result file for two step method does not exist'
    reachable_sets_two_step = pickle.load(open(two_step_pickle_file, "rb"))

    # simulation
    if args.add_random_simulations or args.add_eagerly_searching_simulations:

        network_file_path = f"./models/system_model_1_{args.coeff_p}_{args.coeff_theta}.onnx"
        simulator = Simulator(network_file_path)

        if args.add_random_simulations:
            random_samples = 5000
            p_random = np.random.uniform(p_lb, p_ub, size=(random_samples, 1)).astype(np.float32)
            theta_random = np.random.uniform(theta_lb, theta_ub, size=(random_samples, 1)).astype(np.float32)

        if args.add_eagerly_searching_simulations:
            try:
                samples_path = os.path.join(results_dir, "samples_list.pkl")
                samples_list = pickle.load(open(samples_path, "rb"))
                p_eager = samples_list[0][:, 0].reshape(-1, 1)
                theta_eager = samples_list[0][:, 1].reshape(-1, 1)
            except: 
                samples_list = []
                eager_samples = 5000
                p_eager = np.random.uniform(p_lb, p_ub, size=(eager_samples, 1)).astype(np.float32)
                theta_eager = np.random.uniform(theta_lb, theta_ub, size=(eager_samples, 1)).astype(np.float32)
                samples_list.append(np.hstack((p_eager, theta_eager)))

    p_bounds = np.array([p_lb, p_ub], dtype=np.float32)
    theta_bounds = np.array([theta_lb, theta_ub], dtype=np.float32)
    reachable_cells_baseline = baseline_verifier.get_overlapping_cells_from_intervals(p_bounds, theta_bounds, return_indices=True)
    reachable_cells_one_step_method = reachable_cells_baseline.copy()
    reachable_cells_two_step_method = reachable_cells_baseline.copy()

    global_reachable_sets_baseline |= reachable_cells_baseline
    global_reachable_sets_one_step |= reachable_cells_one_step_method
    global_reachable_sets_two_step |= reachable_cells_two_step_method
    
    current_results_dir = os.path.join(results_dir, "0s") 
    if not os.path.exists(current_results_dir):
        os.makedirs(current_results_dir, exist_ok=True)
    data = {}
    data['reachable_cells_baseline'] = reachable_cells_baseline
    data['reachable_cells_one_step_method'] = reachable_cells_one_step_method
    data['reachable_cells_two_step_method'] = reachable_cells_two_step_method
    data['simulation_p_random'] = p_random
    data['simulation_theta_random'] = theta_random
    if args.add_eagerly_searching_simulations:
        data['simulation_p_eager'] = p_eager
        data['simulation_theta_eager'] = theta_eager
    pickle.dump(data, open(os.path.join(current_results_dir, "data.pkl"), "wb"))
    
    # 0s
    plotter = Plotter(p_lbs, theta_lbs)
    plotter.add_cells(reachable_cells_baseline, color='teal', filled=True, label=f"Baseline ({len(reachable_cells_baseline)} cells)")
    plotter.add_cells(reachable_cells_one_step_method, color='blue', filled=True, label=f"1-step method ({len(reachable_cells_one_step_method)} cells)")
    plotter.add_cells(reachable_cells_two_step_method, color='yellow', filled=True, label=f"2-step method ({len(reachable_cells_two_step_method)} cells)")

    if args.add_random_simulations:
        plotter.add_simulations(p_random, theta_random, color='red', label="Simulations (random samples)")
    if args.add_eagerly_searching_simulations:
        plotter.add_simulations(p_eager, theta_eager, color='black', label="Simulations (eagerly searching samples)")
    image_file_path = os.path.join(current_results_dir, "0s.png")
    plotter.add_safety_violation_region()
    plotter.save_figure(image_file_path, x_range=(-11.0, 11.0), y_range=(-25.0, 25.0))

    step = 1
    while True:
        print(f"Computing reachable cells at step {step}")
        reachable_cells_baseline_ = set()
        reachable_cells_one_step_method_ = set()
        reachable_cells_two_step_method_ = set()

        if args.add_random_simulations:
            print(f"    Simulating next state for random samples")
            p_random, theta_random = simulator.simulate_next_step(p_random, theta_random)
        if args.add_eagerly_searching_simulations:
            print(f"    Simulating next state for eagerly searching samples")
            try:
                p_eager = samples_list[step][:, 0].reshape(-1, 1)
                theta_eager = samples_list[step][:, 1].reshape(-1, 1)
            except:
                p_eager, theta_eager = simulator.simulate_next_step_eagerly_searching(p_eager, theta_eager)
                samples_list.append(np.hstack((p_eager, theta_eager)))

        print(f"    Computing next reachable cells using baseline method")
        for cell in reachable_cells_baseline:
            p_idx, theta_idx = cell
            next_reachable_cells_baseline_1 = baseline_verifier.compute_next_reachable_cells(p_idx, theta_idx, return_indices=True)['reachable_cells']
            next_reachable_cells_baseline = reachable_sets_baseline[p_idx, theta_idx]
            if (-3, -3) in next_reachable_cells_baseline:
                next_reachable_cells_baseline.remove((-3, -3))
            if (-2, -2) in next_reachable_cells_baseline:
                print(p_idx, theta_idx)
                print(next_reachable_cells_baseline)
                print(next_reachable_cells_baseline_1)
                raise Exception("Error occurs, (-2, -2) in the next reachable cells")
            reachable_cells_baseline_ |= next_reachable_cells_baseline
        global_reachable_sets_baseline |= reachable_cells_baseline_
        print(f"len of global reachable_sets using baseline method at step {step} is {len(global_reachable_sets_baseline)}.")
        
        if reachable_cells_baseline_ == reachable_cells_baseline:
            print(f"baseline method has converged at step {step}")
        

        print(f"    Computing next reachable cells using one step method") 
        for cell in reachable_cells_one_step_method:
            p_idx, theta_idx = cell
            next_reachable_cells_one_step = reachable_sets_one_step[p_idx, theta_idx]
            if (-3, -3) in next_reachable_cells_one_step:
                next_reachable_cells_one_step.remove((-3, -3))
            if (-2, -2) in next_reachable_cells_one_step:
                raise Exception("Error occurs, (-2, -2) in the next reachable cells")

            reachable_cells_one_step_method_ |= next_reachable_cells_one_step
        global_reachable_sets_one_step |= reachable_cells_one_step_method_
        print(f"len of global reachable_sets using one step method at step {step} is {len(global_reachable_sets_one_step)}.")

        if reachable_cells_one_step_method_ == reachable_cells_one_step_method:
            print(f"one step method has converged at step {step}")
        
        print(f"    Computing next reachable cells using two step method")
        for cell in reachable_cells_two_step_method:
            p_idx, theta_idx = cell
            if step % 2 != 0:
                next_reachable_cells_two_step = reachable_sets_one_step[p_idx, theta_idx]
            else:
                next_reachable_cells_two_step = reachable_sets_two_step[p_idx, theta_idx]
            if (-3, -3) in next_reachable_cells_two_step:
                next_reachable_cells_two_step.remove((-3, -3))
            if (-2, -2) in next_reachable_cells_two_step:
                raise Exception("Error occurs, (-2, -2) in the next reachable cells")

            reachable_cells_two_step_method_ |= next_reachable_cells_two_step
        global_reachable_sets_two_step |= reachable_cells_two_step_method_
        print(f"len of global reachable_sets using two step method at step {step} is {len(global_reachable_sets_two_step)}.")
        if step % 2 == 0:
            if reachable_cells_two_step_method_ == reachable_cells_two_step_method:
                print(f"two step method has converged at step {step}")

        if step % 2 == 0 \
            and reachable_cells_baseline_ == reachable_cells_baseline \
                and reachable_cells_one_step_method_ == reachable_cells_one_step_method \
                    and reachable_cells_two_step_method_ == reachable_cells_two_step_method:

            print(f"all methods have converged at step {step}")
            current_results_dir = os.path.join(results_dir, f"converged") 
            if not os.path.exists(current_results_dir):
                os.makedirs(current_results_dir, exist_ok=True)
            data = {}
            data['reachable_cells_baseline'] = reachable_cells_baseline
            data['reachable_cells_one_step_method'] = reachable_cells_one_step_method
            data['reachable_cells_two_step_method'] = reachable_cells_two_step_method
            data['simulation_p_random'] = p_random
            data['simulation_theta_random'] = theta_random
            if args.add_eagerly_searching_simulations:
                data['simulation_p_eager'] = p_eager
                data['simulation_theta_eager'] = theta_eager
            pickle.dump(data, open(os.path.join(current_results_dir, "data.pkl"), "wb"))

            plotter = Plotter(p_lbs, theta_lbs)
            plotter.add_cells(reachable_cells_baseline, color='teal', filled=True, label=f"Baseline ({len(reachable_cells_baseline)} cells)")
            plotter.add_cells(reachable_cells_one_step_method, color='blue', filled=True, label=f"1-step method ({len(reachable_cells_one_step_method)} cells)")
            plotter.add_cells(reachable_cells_two_step_method, color='yellow', filled=True, label=f"2-step method ({len(reachable_cells_two_step_method)} cells)")
            if args.add_random_simulations:
                plotter.add_simulations(p_random, theta_random, color='red', label="Simulations (random samples)")
            if args.add_eagerly_searching_simulations:
                plotter.add_simulations(p_eager, theta_eager, color='black', label="Simulations (eagerly searching samples)")
            image_file_path = os.path.join(current_results_dir, "converged.png")
            plotter.add_safety_violation_region()
            plotter.save_figure(image_file_path, x_range=(-11.0, 11.0), y_range=(-25.0, 25.0))

            # save the global reachable sets for one step method using for two step method
            #pickle.dump(global_reachable_sets_one_step, open(os.path.join(results_dir, "global_reachable_sets_one_step.pkl"), "wb"))
            #plotter = Plotter(p_lbs, theta_lbs)
            #plotter.add_cells(global_reachable_sets_one_step, color='blue', filled=True)
            #plotter.save_figure(os.path.join(results_dir, "global_reachable_sets_one_step.png"), x_range=(-11.0, 11.0), y_range=(-25.0, 25.0))
            break

        current_results_dir = os.path.join(results_dir, f"{step}s") 
        if not os.path.exists(current_results_dir):
            os.makedirs(current_results_dir, exist_ok=True)
        data = {}
        data['reachable_cells_baseline'] = reachable_cells_baseline
        data['reachable_cells_one_step_method'] = reachable_cells_one_step_method
        data['reachable_cells_two_step_method'] = reachable_cells_two_step_method
        data['simulation_p_random'] = p_random
        data['simulation_theta_random'] = theta_random
        if args.add_eagerly_searching_simulations:
            data['simulation_p_eager'] = p_eager
            data['simulation_theta_eager'] = theta_eager
        pickle.dump(data, open(os.path.join(current_results_dir, "data.pkl"), "wb"))

        plotter = Plotter(p_lbs, theta_lbs)
        plotter.add_cells(reachable_cells_baseline_, color='teal', filled=True, label=f"Baseline ({len(reachable_cells_baseline_)} cells)")
        plotter.add_cells(reachable_cells_one_step_method_, color='blue', filled=True, label=f"1-step method ({len(reachable_cells_one_step_method_)} cells)")
        plotter.add_cells(reachable_cells_two_step_method_, color='yellow', filled=True, label=f"2-step method ({len(reachable_cells_two_step_method_)} cells)")
        if args.add_random_simulations:
            plotter.add_simulations(p_random, theta_random, color='red', label="Simulations (random samples)")
        if args.add_eagerly_searching_simulations:
            plotter.add_simulations(p_eager, theta_eager, color='black', label="Simulations (optimized samples)")
        image_file_path = os.path.join(current_results_dir, f"{step}s.png")
        plotter.add_safety_violation_region()
        plotter.save_figure(image_file_path, x_range=(-11.0, 11.0), y_range=(-25.0, 25.0))

        reachable_cells_baseline = reachable_cells_baseline_
        reachable_cells_one_step_method = reachable_cells_one_step_method_
        if step % 2 == 0:
            reachable_cells_two_step_method = reachable_cells_two_step_method_
        step += 1


if __name__ == '__main__':
    main()