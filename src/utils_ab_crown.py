import numpy as np
import math
from collections import defaultdict
import abcrown, arguments
import torch
from load_model import load_model
import os
import subprocess
import yaml

import time
import pickle

import logging

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / 1024**2

def save_vnnlib(input_bounds, mid, sign, spec_path="./temp.vnnlib"):

    with open(spec_path, "w") as f:

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        f.write(f"(declare-const Y_0 Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write(f"(assert ({sign} Y_0 {mid}))\n")

class MultiStepVerifier:
    def __init__(self, p_lbs, p_ubs, theta_lbs, theta_ubs, step=1, latent_bounds=0.8, simulation_samples=10000, reachable_cells_path=None, p_coeff=-0.74, theta_coeff=-0.44) -> None:

        self.p_lbs = p_lbs
        self.p_ubs = p_ubs
        self.theta_lbs = theta_lbs
        self.theta_ubs = theta_ubs
        #self.p_safe_bounds = p_safe_bounds
        #self.theta_safe_bounds = theta_safe_bounds
        #self.p_safe_lb_idx = math.ceil((p_safe_bounds[0] - p_lbs[0])/(p_ubs[0] - p_lbs[0]))
        #self.p_safe_ub_idx = math.floor((p_safe_bounds[1] - p_ubs[0])/(p_ubs[0] - p_lbs[0]))
        #self.theta_safe_lb_idx = math.ceil((theta_safe_bounds[0] - theta_lbs[0])/(theta_ubs[0] - theta_lbs[0]))
        #self.theta_safe_ub_idx = math.floor((theta_safe_bounds[1] - theta_ubs[0])/(theta_ubs[0] - theta_lbs[0]))

        if reachable_cells_path is not None:
            self.reachable_cells = pickle.load(open(reachable_cells_path, 'rb'))
        else:
            self.simulation_samples = simulation_samples
            self.step = step
            self.latent_bounds = latent_bounds
            self.p_coeff = p_coeff
            self.theta_coeff = theta_coeff
        
            arguments.Config.all_args['model']['input_shape'] = [-1, 2 + step * 2]
            arguments.Config.all_args['model']['path'] = f'./models/single_step_{self.p_coeff}_{self.theta_coeff}.pth'

    def load_abcrown_setting(self, setting_idx, output_idx, num_steps, vnnpath, spec_path="./temp.vnnlib"):
        # setting_idx: 0, 1, 2, 3, 4, 5, 6, 7
        settings = {
            'general': {
                'device': 'cuda',
                #'conv_mode': 'matrix',
                'results_file': 'results.txt',
            },

            'model': {
                'name': f'Customized("custom_model_data", "MultiStepTaxiNet", index={output_idx}, num_steps={num_steps})',
                'path': f'./models/single_step_{self.p_coeff}_{self.theta_coeff}.pth',
                'input_shape': [-1, 2 + num_steps * 2],
            },
            
            'data': {
                'dataset': 'CGAN',
                'num_outputs': 1,
            },
            
            'specification': {
                'vnnlib_path': f"{vnnpath}",
            },

            'solver': {
                'batch_size': 1,
                'auto_enlarge_batch_size': True,
            },

            'attack': {
                'pgd_order': 'before',
                'pgd_restarts': 100
            },

            'bab': {
                'initial_max_domains': 100,
                'branching': {
                    'method': 'sb',
                    'sb_coeff_thresh': 0.01,
                    'input_split': {
                        'enable': True,
                        'catch_assertion': True,
                    },
                },
                'timeout': 300,
            }
        }
        
        batch_size = [4096, 1024, 512, 128, 32, 16, 8]
        if setting_idx == 0:
            settings['general']['enable_incomplete_verification'] = True
            settings['solver']['bound_prop_method'] = 'crown'
            settings['solver']['crown'] = {'batch_size': 512}
            
        else:
            # try complete verification with alpha-crown
            settings['general']['enable_incomplete_verification'] = False
            settings['solver']['bound_prop_method'] = 'alpha-crown'
            settings['solver']['crown'] = {'batch_size' : batch_size[setting_idx - 1]}
            settings['solver']['alpha-crown'] = {'lr_alpha': 0.25, 'iteration': 20, 'full_conv_alpha': False}
            settings['solver']['beta-crown'] = {'lr_alpha': 0.05, 'lr_beta': 0.1, 'iteration': 5}
        
        with open(spec_path, 'w') as f:
            yaml.dump(settings, f)

    def read_results(self, result_path):
        if os.path.exists(result_path):
            with open(result_path, "rb") as f:
                lines = pickle.load(f)
                results = lines['results'][0][0]
            return results

        else:
            return "unknown"

    def check_property(self, init_box, mid, sign, idx):
        neg_sign = "<=" if sign == ">=" else ">="
        spec_path = "./temp.vnnlib"
        config_path = "./cgan.yaml"
        result_path = "./results.txt"
        ## clean the file first
        if os.path.exists(spec_path):
            os.remove(spec_path)
        assert not os.path.exists(spec_path)
        save_vnnlib(init_box, mid, neg_sign, spec_path="./temp.vnnlib")

        for setting_idx in range(8): # try different settings, starting from incomplete verification, and then complete verification with different batch sizes
            time_start_check = time.time()
            if os.path.exists(result_path):
                os.remove(result_path)
            if os.path.exists(config_path):
                os.remove(config_path)
            assert not os.path.exists(config_path)
            assert not os.path.exists(result_path)

            self.load_abcrown_setting(setting_idx, output_idx=idx, num_steps=self.step, vnnpath=spec_path, spec_path=config_path)

            logging.info(f"                using setting {setting_idx}")
            logging.info(f"                gpu memory usage: {get_gpu_memory_usage()}")

            ### verify the property
            #verified_status = abcrown.main()

            ### using subprocess to run the verification
            ### this is because the abcrown will change the gpu memory usage, and we need to clear the memory after each verification
            tool_dir = os.environ.get('TOOL_DIR')
            process = subprocess.Popen(["python", 
                                        os.path.join(tool_dir, "complete_verifier/abcrown.py"),
                                        "--config",
                                        "./cgan.yaml"])
            process.wait()

            verified_status = self.read_results(result_path)
            time_end_check = time.time()

            logging.info(f"                verification status: {verified_status}, taking {time_end_check - time_start_check} seconds")
            if verified_status != "unknown" and verified_status != "timed out":
                break
            else:
                logging.info(f"                setting {setting_idx} failed, taking {time_end_check - time_start_check} seconds")

        self.num_calls_alpha_beta_crown += 1

        if verified_status not in ["safe", "safe-incomplete", "unsafe-pgd", "unsafe-bab", "safe-incomplete (timed out)"]:
            self.setting_idx_for_each_call.append(-1)
            self.time_for_each_call.append(-1)
        else:
            self.setting_idx_for_each_call.append(setting_idx)
            self.time_for_each_call.append(time_end_check - time_start_check)

        if verified_status == "unsafe-pgd" or verified_status == "unsafe-bab":
            return False
        elif verified_status == "safe" or verified_status == "safe-incomplete" or verified_status == "safe-incomplete (timed out)":
            return True
        elif verified_status == "unknown" or verified_status == "timed out":
            return None
        else:
            raise NotImplementedError(f"The verification status {verified_status} is not implemented")

    def get_overlapping_cells(self, lb_ub, ub_lb, init_box, index):
        # lb_ub: the lower bound's upper bound
        # ub_lb: the upper bound's lower bound
        # init_box: the initial box
        # index: the index of the variable to be searched

        if index == 0:
            # the index of the variable to be searched is p
            # the lb_ub should be greater than or equal to the lb of the p
            # the ub_ should be less than or equal to the ub of the p
            # the unsafe case should be handled outside of this function
            assert lb_ub >= self.p_lbs[0] and ub_lb <= self.p_ubs[-1]
        
        ubs = self.p_ubs if index == 0 else self.theta_ubs
        lbs = self.p_lbs if index == 0 else self.theta_lbs
        
        ## search the lb
        logging.info(f"        search for lb for idx {index}")
        right_idx = math.floor((lb_ub - lbs[0])/(ubs[0]-lbs[0]))
        found_lb = False

        if right_idx < 0:
            # for p, wont happen
            # for theta, might happen, which means the lb_idx is -1
            lb_idx = -1
            found_lb = True
        
        else:
            if right_idx >= len(lbs):
                # for p, wont happen
                # for theta, might happen, which means the lb might be greater than the ub of the input
                # we should check if the lb is greater
                logging.info(f"            checking output >= {ubs[-1]}: {len(ubs)}")
                result = self.check_property(init_box, ubs[-1], ">=", index)

                if result == None:
                # if the error occurs, we set the lower bound to len(lbs)-1, this is over-approximation
                # example: if the real lb is greater than the up_bound of the input, 
                # however, we cannot prove it due to error, we use idx=len(lbs)-1 as the lower bound
                # we should continue the search for the lower bound
                    logging.info(f"            error occurs when checking output >= {ubs[-1]}, set the lower bound to {len(lbs)-1}")
                    self.error_during_verification = True

                else: 
                    if result:
                        logging.info(f"            verified, the lb is out of the range")
                        return (-1, -1)
                    else:
                        logging.info(f"            the lb is not guaranteed greater or equal to {ubs[-1]}, set the lower bound to {len(lbs)-1}")

                right_idx = len(lbs)-1
            
            for i in range(right_idx, -1, -1):
                logging.info(f"            checking output >= {lbs[i]}: {i}")
                result = self.check_property(init_box, lbs[i], ">=", index)

                if result == None:
                    self.error_during_verification = True


                    #error_found_lb = True
                    logging.info(f"            error occurs when checking output >= {lbs[i]}: {i}")
                    continue

                if result:
                    logging.info(f"            verified, the lb idx is {i}")
                    found_lb = True
                    lb_idx = i
                    break

            if not found_lb:
                logging.info(f"            the lb is not guaranteed greater or equal to {lbs[0]}")
                lb_idx = -1

                # sometimes the error not occurs during verification but the lb is still not found
                # example: the lb might be less than lbs[0], the lb checking will only check till the "lb >= lbs[0]", and then skip.
                # if there is no error, it means the lb < 0.0
                # else the error occurs, it means the lb might be greater than 0.0
                #if error_found_lb:
                #    logging.info(f"            this bound cannot be verified due to error, return -2")
                #    lb_idx = -2
        
        ## search the ub
        logging.info(f"        search for ub for idx {index}")
        left_idx = math.ceil((ub_lb - ubs[0])/(ubs[0]-lbs[0]))
        found_ub = False
        if left_idx >= len(ubs):
            # for p, wont happen
            # for theta, might happen, which means the ub_idx is len(ubs)
            ub_idx = len(ubs)
            found_ub = True

        else:
            if left_idx < 0:
                # for p, wont happen
                # for theta, might happen, which means the ub might be less than the lb of the input
                # we should check if the ub is less
                logging.info(f"            checking output <= {lbs[0]}")
                result = self.check_property(init_box, lbs[0], "<=", index)

                if result == None:
                # if the error occurs, we set the upper bound to 0, this is over-approximation
                # example: if the real ub is less than the lower_bound of the input, 
                # however, we cannot prove it due to error, we use idx=0 as the upper bound
                # we should continue the search for the upper bound
                    logging.info(f"            error occurs when checking output <= {lbs[0]}, set the upper bound to 0")
                    self.error_during_verification = True

                else: 
                    if result:
                        logging.info(f"            verified, the ub is out of the range")
                        return (-1, -1)
                    else:
                        logging.info(f"            the ub is not guaranteed less or equal to {lbs[0]}, set the upper bound to 0")
                
                left_idx = 0
            
            for i in range(left_idx, len(ubs)):
                logging.info(f"            checking output <= {ubs[i]}: {i}")
                result = self.check_property(init_box, ubs[i], "<=", index)

                if result == None:
                    self.error_during_verification = True
                    logging.info(f"            error occurs when checking output <= {ubs[i]}: {i}")
                    continue

                if result:
                    logging.info(f"            verified, the ub idx is {i}")
                    found_ub = True
                    ub_idx = i
                    break
            
            if not found_ub:
                logging.info(f"            the ub is not guaranteed less or equal to {ubs[-1]}")
                ub_idx = len(ubs)
        
        logging.info(f"        lb_idx: {lb_idx}, ub_idx: {ub_idx}") 
        return (lb_idx, ub_idx)

    def get_intervals(self, p_idx, theta_idx):
        p_lb = self.p_lbs[p_idx]
        p_ub = self.p_ubs[p_idx]
        theta_lb = self.theta_lbs[theta_idx]
        theta_ub = self.theta_ubs[theta_idx]

        init_box = [[p_lb, p_ub],
                    [theta_lb, theta_ub]]

        init_box.extend([[-self.latent_bounds, self.latent_bounds]]*2*self.step)
        init_box = np.array(init_box, dtype=np.float32)


        # simulate the system
        samples = self.simulation_samples
        inputs = []
        for bounds in init_box:
            inputs.append(np.random.uniform(bounds[0], bounds[1], samples).astype(np.float32))
        inputs = np.stack(inputs, axis=1)

        # p
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStepTaxiNet", index=0, num_steps={self.step})'
        # in order to save the gpu memory, we load the model for each simulation
        model_ori = load_model().cuda()
        model_ori.eval()
        outputs = model_ori(torch.from_numpy(inputs).cuda())
        
        del model_ori
        torch.cuda.empty_cache()
        p_lb_sim = torch.min(outputs).item()
        p_ub_sim = torch.max(outputs).item()
        logging.info(f"    p_lb_sim: {p_lb_sim}, p_ub_sim: {p_ub_sim}")

        del outputs
        torch.cuda.empty_cache()
        
        #if p out of the boundary, return (-1, 0, 0, 0)
        if p_lb_sim <= self.p_lbs[0] or p_ub_sim >= self.p_ubs[-1]:
            # the aircraft is already out of the track, return (-1, 0, 0, 0)
            return (-1, 0, 0, 0), (-1, 0, 0, 0)


        ## [p_sim_lb_idx, p_sim_ub_idx)
        p_sim_lb_idx = math.floor((p_lb_sim - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0]))
        p_sim_ub_idx = math.ceil((p_ub_sim - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0]))

        p_lb_idx, p_ub_idx = self.get_overlapping_cells(p_lb_sim, p_ub_sim, init_box, index=0)
        
        if p_lb_idx == -1 and p_ub_idx == -1:
            # the aircraft is already in the danger zone, return (-1, 0, 0, 0)
            return [-1, 0, 0, 0], [p_sim_lb_idx, p_sim_ub_idx, 0, 0]

        # theta
        arguments.Config.all_args['model']['name'] = f'Customized("custom_model_data", "MultiStepTaxiNet", index=1, num_steps={self.step})'
        # in order to save the gpu memory, we load the model for each simulation
        model_ori = load_model().cuda()
        model_ori.eval()
        outputs = model_ori(torch.from_numpy(inputs).cuda())

        del model_ori
        torch.cuda.empty_cache()
        theta_lb_sim = torch.min(outputs).item()
        theta_ub_sim = torch.max(outputs).item()
        logging.info(f"    theta_lb_sim: {theta_lb_sim}, theta_ub_sim: {theta_ub_sim}")

        del outputs
        torch.cuda.empty_cache()

        # if theta out of the boundary, we dont need to do anything

        ## [theta_sim_lb_idx, theta_sim_ub_idx)
        theta_sim_lb_idx = math.floor((theta_lb_sim - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0]))
        theta_sim_ub_idx = math.ceil((theta_ub_sim - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0]))

        theta_lb_idx, theta_ub_idx = self.get_overlapping_cells(theta_lb_sim, theta_ub_sim, init_box, index=1)

        if theta_lb_idx == -1 and theta_ub_idx == -1:
            # the theta of aircraft is always out of the boundary
            return [p_lb_idx, p_ub_idx, -1, -1], [p_sim_lb_idx, p_sim_ub_idx, theta_sim_lb_idx, theta_sim_ub_idx]

        return [p_lb_idx, p_ub_idx, theta_lb_idx, theta_ub_idx], [p_sim_lb_idx, p_sim_ub_idx, theta_sim_lb_idx, theta_sim_ub_idx]


    def compute_next_reachable_cells(self, p_idx, theta_idx):
        result_dict = dict()

        if hasattr(self, 'reachable_cells'):
            result_dict["reachable_cells"] = self.reachable_cells[(p_idx, theta_idx)]
            return result_dict
        
        time_dict = dict()
        self.num_calls_alpha_beta_crown = 0
        self.setting_idx_for_each_call = []
        self.time_for_each_call = [] # this time is the time for each verification using the successful setting
        self.error_during_verification = False

        time_start = time.time()
        interval, sim_interval = self.get_intervals(p_idx, theta_idx)
        time_end_get_intervals = time.time()
        logging.info(f"    Interval index: {interval}")
        
        result_dict["sim_interval"] = sim_interval
        if interval[0] == -1:
            ## this means unsafe, p is out of the range
            result_dict["reachable_cells"] = {(-2, -2)}
        
        elif interval[2] == -1 and interval[3] == -1:
            ## this means theta is out of the range
            result_dict["reachable_cells"] = {(-3, -3)}
        
        else:
            reachable_cells = set()

            ## first check if p is safe
            assert 0<= interval[0] <= interval[1] < len(self.p_lbs)

            # filter out theta out of the range
            if interval[2] < 0:
                reachable_cells.add((-3, -3))
                interval[2] = 0

            if interval[3] >= len(self.theta_lbs):
                reachable_cells.add((-3, -3))
                interval[3] = len(self.theta_lbs) - 1
            
            assert 0<= interval[2] <= interval[3] < len(self.theta_lbs)

            for p_idx in range(interval[0], interval[1]+1):
                for theta_idx in range(interval[2], interval[3]+1):
                    reachable_cells.add((p_idx, theta_idx))
            result_dict['reachable_cells'] = reachable_cells

        time_end = time.time()
        time_dict["whole_time"] = time_end - time_start
        time_dict["get_intervals_time"] = time_end_get_intervals - time_start
        result_dict["time"] = time_dict
        result_dict["num_calls_alpha_beta_crown"] = self.num_calls_alpha_beta_crown
        result_dict["setting_idx_for_each_call"] = self.setting_idx_for_each_call
        result_dict["verification_time"] = self.time_for_each_call
        result_dict["error_during_verification"] = self.error_during_verification

        return result_dict