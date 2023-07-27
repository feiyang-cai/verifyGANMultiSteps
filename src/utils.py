import numpy as np
import math
import csv
from collections import defaultdict

from nnenum import nnenum
from nnenum.settings import Settings
from nnenum.util import compress_init_box
from nnenum.lp_star import LpStar
import onnxruntime as ort

from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import time

class Plotter:
    def __init__(self, p_lbs, theta_lbs) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8,8), dpi=200)
        self.p_bounds = [np.inf, -np.inf]
        self.theta_bounds = [np.inf, -np.inf]
        self.p_lbs = p_lbs
        self.theta_lbs = theta_lbs
        self.cell_width = (p_lbs[1] - p_lbs[0])
        self.cell_height = (theta_lbs[1] - theta_lbs[0])
        self.legend_label_list = []
        self.legend_list = []
    

    def add_patches(self, patches, color, label=None):
        for patch in patches:
            x = patch[0]
            y = patch[2]
            width = patch[1] - patch[0]
            height = patch[3] - patch[2]
            rec = plt.Rectangle((x, y), width, height, color=color, alpha=1.0)
            self.ax.add_patch(rec)
            self.p_bounds[0] = min(self.p_bounds[0], x)
            self.p_bounds[1] = max(self.p_bounds[1], x+width)
            self.theta_bounds[0] = min(self.theta_bounds[0], y)
            self.theta_bounds[1] = max(self.theta_bounds[1], y+height)

        if label is not None:
            self.legend_label_list.append(label)
            self.legend_list.append(rec)
    
    def add_cells(self, cells, color, label=None, filled=False):
        for cell in cells:
            x = self.p_lbs[cell[0]]
            y = self.theta_lbs[cell[1]]
            cell = plt.Rectangle((x, y), self.cell_width, self.cell_height, fill=filled, linewidth=2, edgecolor=color, alpha=1)
            self.ax.add_patch(cell)
            self.p_bounds[0] = min(self.p_bounds[0], x)
            self.p_bounds[1] = max(self.p_bounds[1], x+self.cell_width)
            self.theta_bounds[0] = min(self.theta_bounds[0], y)
            self.theta_bounds[1] = max(self.theta_bounds[1], y+self.cell_height)

        if label is not None:
            self.legend_label_list.append(label)
            self.legend_list.append(cell)
    
    def add_verts(self, verts, color, label=None):
        for vert in verts:
            if len(vert) < 2:
                continue
            codes = [Path.MOVETO] + [Path.LINETO] * (len(vert) - 2) + [Path.CLOSEPOLY]
            path = Path(vert, codes)
            patch = patches.PathPatch(path, edgecolor='None', facecolor=color, alpha=1.0)
            self.ax.add_patch(patch)

        if label is not None:
            self.legend_label_list.append(label)
            self.legend_list.append(patch)
    
    def add_simulations(self, ps, thetas, color, label=None):
        scatter = self.ax.scatter(ps, thetas, c=color, alpha=0.8, s=1)
        if label is not None:
            self.legend_label_list.append(label)
            self.legend_list.append(scatter)
    
    def save_figure(self, file_name, x_range=None, y_range=None):
        if x_range is not None and y_range is not None:
            self.ax.set_xlim(x_range[0], x_range[1])
            self.ax.set_ylim(y_range[0], y_range[1])
        else:    
            self.ax.set_xlim(self.p_bounds[0]-0.2, self.p_bounds[1]+0.2)
            self.ax.set_ylim(self.theta_bounds[0]-0.2, self.theta_bounds[1]+0.2)

        ## plot grids
        for p_lb in self.p_lbs:
            X = [p_lb, p_lb]
            Y = [self.theta_lbs[0], self.theta_lbs[-1]]
            self.ax.plot(X, Y, 'lightgray', alpha=0.2)

        for theta_lb in self.theta_lbs:
            Y = [theta_lb, theta_lb]
            X = [self.p_lbs[0], self.p_lbs[-1]]
            self.ax.plot(X, Y, 'lightgray', alpha=0.2)
        
        
        self.ax.set_xlabel(r"$p$ (m)")
        self.ax.set_ylabel(r"$\theta$ (degrees)")
        if len(self.legend_list) != 0:
            self.ax.legend(self.legend_list, self.legend_label_list, loc='lower right')
        self.fig.savefig(file_name)
        plt.close()

class Simulator:
    def __init__(self, network_file_path) -> None:

        shared_library = "libcustom_dynamics.so"
        so = ort.SessionOptions()
        so.register_custom_ops_library(shared_library)

        self.session = ort.InferenceSession(network_file_path, sess_options=so)
    
    def simulate_next_step(self, ps, thetas):
        samples = len(ps)
        zs = np.random.uniform(-0.8, 0.8, size=(samples, 2)).astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_name = self.session.get_outputs()[0].name

        p_list = []
        theta_list = []
        for z_i, p_i, theta_i in zip(zs, ps, thetas):
            input_0 = np.concatenate([z_i, p_i, theta_i]).astype(np.float32).reshape(input_shape)
            res = self.session.run([output_name], {input_name: input_0})
            p_, theta_ = res[0][0]
            p_list.append(p_)
            theta_list.append(theta_)
        return np.array(p_list).reshape(-1, 1), np.array(theta_list).reshape(-1, 1)


class BaselineVerifier:
    def __init__(self, network_file_path, p_lbs, p_ubs, theta_lbs, theta_ubs, csv_file=None) -> None:
        self.network = nnenum.load_onnx_network(network_file_path)
        self.p_lbs = p_lbs
        self.p_ubs = p_ubs
        self.theta_lbs = theta_lbs
        self.theta_ubs = theta_ubs
        if csv_file is not None:
            self.control_bounds = defaultdict(tuple)
            with open(csv_file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    cell = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                    self.control_bounds[cell] = (float(row[4]), float(row[5]))

    def get_control_interval_bounds_from_stars(self, stars):
        """
        Get the control interval bounds from the stars.
        :param stars: list of stars
        :return: control interval bounds
        """
        control_interval_bounds = [np.inf, -np.inf]
        for star in stars:
            u_ub = star.minimize_output(4, True)
            u_lb = star.minimize_output(4, False)

            control_interval_bounds[0] = min(control_interval_bounds[0], u_lb)
            control_interval_bounds[1] = max(control_interval_bounds[1], u_ub)


        return control_interval_bounds

    def dynamics(self, control_bound, p_bound, theta_bound, steps=20, radians=False):
        v, L, dt = 5, 5, 0.05
        (control_lb, control_ub) = control_bound
        (p_lb, p_ub) = p_bound
        (theta_lb, theta_ub) = theta_bound

        for step in range(steps): # 1s
            if radians:
                p_lb = p_lb + v*dt*math.sin(theta_lb)
                p_ub = p_ub + v*dt*math.sin(theta_ub)
                theta_lb = theta_lb + dt * v/L*math.tan(control_lb)
                theta_ub = theta_ub + dt * v/L*math.tan(control_ub)
            else:
                p_lb = p_lb + v*dt*math.sin(math.radians(theta_lb))
                p_ub = p_ub + v*dt*math.sin(math.radians(theta_ub))
                theta_lb = theta_lb + dt * math.degrees(v/L*math.tan(math.radians(control_lb)))
                theta_ub = theta_ub + dt * math.degrees(v/L*math.tan(math.radians(control_ub)))

        return (p_lb, p_ub), (theta_lb, theta_ub)
    
    def get_overlapping_cells_from_intervals(self, p_bounds, theta_bounds, return_indices=False):
        """
        Get the overlapping cells from the intervals.
        :param p_bounds: position bounds
        :param theta_bounds: angle bounds
        :return: overlapping cells
        """

        reachable_cells = set()

        # get the lower and upper bound indices of the output interval
        p_lb_idx = math.floor((p_bounds[0] - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # floor
        p_ub_idx = math.ceil((p_bounds[1] - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # ceil

        theta_lb_idx = math.floor((theta_bounds[0] - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # floor
        theta_ub_idx = math.ceil((theta_bounds[1] - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # ceil

        assert 0<=p_lb_idx<len(self.p_lbs)
        assert 1<=p_ub_idx<=len(self.p_ubs)
        assert 0<=theta_lb_idx<len(self.theta_lbs)
        assert 1<=theta_ub_idx<=len(self.theta_ubs)

        for p_idx in range(p_lb_idx, p_ub_idx):
            for theta_idx in range(theta_lb_idx, theta_ub_idx):
                if return_indices:
                    reachable_cells.add((p_idx, theta_idx))
                else:
                    reachable_cells.add((self.p_lbs[p_idx], self.p_ubs[p_idx], 
                                         self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]))

        return reachable_cells
    
    def compute_next_reachable_cells(self, p_idx, theta_idx, return_indices=False, return_bounds=False):

        # set nneum settings
        nnenum.set_exact_settings()
        Settings.GLPK_TIMEOUT = 10
        Settings.PRINT_OUTPUT = False
        Settings.TIMING_STATS = True
        Settings.RESULT_SAVE_STARS = True
        Settings.SPLIT_TOLERANCE = 1e-3

        p_lb = self.p_lbs[p_idx]
        p_ub = self.p_ubs[p_idx]
        theta_lb = self.theta_lbs[theta_idx]
        theta_ub = self.theta_ubs[theta_idx]

        if hasattr(self, 'control_bounds'):
            control_bounds = self.control_bounds[(p_lb, p_ub, theta_lb, theta_ub)]
        else:
            init_box = [[-0.8, 0.8], [-0.8, 0.8]]
            init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
            init_box = np.array(init_box, dtype=np.float32)
            init_bm, init_bias, init_box = compress_init_box(init_box)
            star = LpStar(init_bm, init_bias, init_box)

            # start to verify the network
            result = nnenum.enumerate_network(star, self.network)
            control_bounds = self.get_control_interval_bounds_from_stars(result.stars)
            control_bounds = np.rad2deg(control_bounds)
        p_bounds_, theta_bounds_ = self.dynamics(control_bounds, (p_lb, p_ub), (theta_lb, theta_ub), radians=False)
        reachable_cells = self.get_overlapping_cells_from_intervals(p_bounds_, theta_bounds_, return_indices=return_indices)

        if return_bounds:
            return reachable_cells, p_bounds_, theta_bounds_
        else:
            return reachable_cells

class MultiStepVerifier:
    def __init__(self, network_file_path, p_lbs, p_ubs, theta_lbs, theta_ubs) -> None:
        Settings.ONNX_WHITELIST.append("TaxiNetDynamics")
        self.network = nnenum.load_onnx_network(network_file_path)
        self.p_lbs = p_lbs
        self.p_ubs = p_ubs
        self.theta_lbs = theta_lbs
        self.theta_ubs = theta_ubs

        shared_library = "libcustom_dynamics.so"
        so = ort.SessionOptions()
        so.register_custom_ops_library(shared_library)

        self.session = ort.InferenceSession(network_file_path, sess_options=so)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.step = (self.input_shape[1]-2)//2
    
    def compute_interval_enclosure(self, star):
        # compute the enclosure of the output interval
        p_ub = star.minimize_output(0, True)
        p_lb = star.minimize_output(0, False)
        theta_ub = star.minimize_output(1, True)
        theta_lb = star.minimize_output(1, False)

        # p of reachable_sets might be out of the range (unsafe), filter them out here
        # filter them out here
        if p_lb < self.p_lbs[0] or p_ub > self.p_ubs[-1]:
            return [[-1, -1], [-1, -1]]

        # get the lower and upper bound indices of the output interval
        p_lb_idx = math.floor((p_lb - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # floor
        p_ub_idx = math.ceil((p_ub - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # ceil

        theta_lb_idx = math.floor((theta_lb - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # floor
        theta_ub_idx = math.ceil((theta_ub - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # ceil

        assert p_lb_idx >= 0 and p_ub_idx <= len(self.p_lbs)

        theta_lb_idx = max(theta_lb_idx, 0)
        theta_ub_idx = min(theta_ub_idx, len(self.theta_lbs))

        return [[p_lb_idx, p_ub_idx], [theta_lb_idx, theta_ub_idx]]

    def check_intersection(self, star, p_idx, theta_idx):
        # get the cell bounds
        p_lb = self.p_lbs[p_idx]
        p_ub = self.p_ubs[p_idx]
        theta_lb = self.theta_lbs[theta_idx]
        theta_ub = self.theta_ubs[theta_idx]

        p_bias = star.bias[0]
        theta_bias = star.bias[1]

        if "ita" not in star.lpi.names:
            p_mat = star.a_mat[0, :]
            theta_mat = star.a_mat[1, :]

            # add the objective variable 'ita'
            star.lpi.add_cols(['ita'])

            # add constraints

            ## p_mat * p - ita <= p_ub - p_bias
            p_mat_1 = np.hstack((p_mat, -1))
            star.lpi.add_dense_row(p_mat_1, p_ub - p_bias)

            ## -p_mat * p - ita <= -p_lb + p_bias
            p_mat_2 = np.hstack((-p_mat, -1))
            star.lpi.add_dense_row(p_mat_2, -p_lb + p_bias)

            ## theta_mat * theta - ita <= theta_ub - theta_bias
            theta_mat_1 = np.hstack((theta_mat, -1))
            star.lpi.add_dense_row(theta_mat_1, theta_ub - theta_bias)

            ## -theta_mat * theta - ita <= -theta_lb + theta_bias
            theta_mat_2 = np.hstack((-theta_mat, -1))
            star.lpi.add_dense_row(theta_mat_2, -theta_lb + theta_bias)

        else:
            rhs = star.lpi.get_rhs()
            rhs[-4] = p_ub - p_bias
            rhs[-3] = -p_lb + p_bias
            rhs[-2] = theta_ub - theta_bias
            rhs[-1] = -theta_lb + theta_bias
            star.lpi.set_rhs(rhs)

        direction_vec = [0] * star.lpi.get_num_cols()
        direction_vec[-1] = 1
        rv = star.lpi.minimize(direction_vec)
        return rv[-1] <= 0.0

    def get_reachable_cells_from_stars(self, stars, reachable_cells, return_indices=False, return_verts=False):
        verts = []
        for star in stars:

            # compute the interval enclosure for the star set to get the candidate cells
            ## TODO: using zonotope enclosure may be faster,
            ## but we need to solve LPs to get the candidate cells
            interval_enclosure = self.compute_interval_enclosure(star)

            ## if p is out of the range (unsafe), then clear the reachable cells
            if interval_enclosure == [[-1, -1], [-1, -1]]:
                reachable_cells = set()
                if return_indices:
                    reachable_cells.add((-2, -2))
                else:
                    reachable_cells.add((-2, -2, -2, -2))
                break

            if return_verts:
                verts.append(star.verts())

            # if the theta out of the range, discard the star
            if interval_enclosure[1][0] >= len(self.theta_lbs) or interval_enclosure[1][1] <= 0:
                if return_indices:
                    reachable_cells.add((-3, -3))
                else:
                    reachable_cells.add((-3, -3, -3, -3))
                continue

            assert interval_enclosure[0][0] <= interval_enclosure[0][1] - 1
            assert interval_enclosure[1][0] <= interval_enclosure[1][1] - 1

            ## if only one candidate cell, then skip
            if interval_enclosure[0][0] == interval_enclosure[0][1] - 1 and interval_enclosure[1][0] == interval_enclosure[1][1] - 1:
                if return_indices:
                    reachable_cells.add((interval_enclosure[0][0], interval_enclosure[1][0]))
                else:
                    reachable_cells.add((self.p_lbs[interval_enclosure[0][0]], self.p_ubs[interval_enclosure[0][0]],
                                         self.theta_lbs[interval_enclosure[1][0]], self.theta_ubs[interval_enclosure[1][0]]))
                continue

            # intersection check for the candidate cells
            for p_idx in range(interval_enclosure[0][0], interval_enclosure[0][1]):
                for theta_idx in range(interval_enclosure[1][0], interval_enclosure[1][1]):
                    if return_indices and (p_idx, theta_idx) not in reachable_cells:
                        if self.check_intersection(star, p_idx, theta_idx):
                            reachable_cells.add((p_idx, theta_idx))
                    elif (not return_indices) and (self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]) not in reachable_cells:
                        if self.check_intersection(star, p_idx, theta_idx):
                            reachable_cells.add(((self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx])))

        if return_verts:
            return reachable_cells, verts
        else:
            return reachable_cells


    def compute_next_reachable_cells(self, p_idx, theta_idx, return_indices=False, return_verts=False, print_output=False, pbar=None, return_tolerance=False, single_thread=False):
        reachable_cells = set()
        time_dict = defaultdict()

        t_start = time.time()

        p_lb = self.p_lbs[p_idx]
        p_ub = self.p_ubs[p_idx]
        theta_lb = self.theta_lbs[theta_idx]
        theta_ub = self.theta_ubs[theta_idx]

        # simulations
        t_start_sim = time.time()
        samples = 5000
        z = np.random.uniform(-0.8, 0.8, size=(samples, self.step*2)).astype(np.float32)
        p = np.random.uniform(p_lb, p_ub, size=(samples, 1)).astype(np.float32)
        theta = np.random.uniform(theta_lb, theta_ub, size=(samples, 1)).astype(np.float32)

        for z_i, p_i, theta_i in zip(z, p, theta):
            assert p_i<=p_ub and p_i>=p_lb
            assert theta_i<=theta_ub and theta_i>=theta_lb
            z_i_pre = z_i[:2]
            z_i_post = z_i[2:]
            input_0 = np.concatenate([z_i_pre, p_i, theta_i, z_i_post]).astype(np.float32).reshape(self.input_shape)
            res = self.session.run([self.output_name], {self.input_name: input_0})
            p_, theta_ = res[0][0]
            p_idx = math.floor((p_ - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # floor
            theta_idx = math.floor((theta_ - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # floor
            if return_indices:
                reachable_cells.add((p_idx, theta_idx))
            else:
                reachable_cells.add((self.p_lbs[p_idx], self.p_ubs[p_idx], 
                                     self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]))

        t_end_sim = time.time()
        time_dict['simulation'] = t_end_sim - t_start_sim

        # set nneum settings
        nnenum.set_exact_settings()
        Settings.GLPK_TIMEOUT = 10
        Settings.PRINT_OUTPUT = print_output
        Settings.TIMING_STATS = False
        Settings.RESULT_SAVE_STARS = True
        if single_thread:
            Settings.NUM_PROCESSES = 1

        init_box = [[-0.8, 0.8], [-0.8, 0.8]]
        init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
        init_box.extend([[-0.8, 0.8]]*((self.step-1)*2))
        init_box = np.array(init_box, dtype=np.float32)
        init_bm, init_bias, init_box = compress_init_box(init_box)
        star = LpStar(init_bm, init_bias, init_box)

        return_dict = dict()

        for split_tolerance in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            t_start_enum = time.time()
            if pbar is not None:
                pbar.set_description(f"split_tolerance={split_tolerance}")
            else:
                print(f"split_tolerance={split_tolerance}")
            Settings.SPLIT_TOLERANCE = split_tolerance # small outputs get rounded to zero when deciding if splitting is possible
            result = nnenum.enumerate_network(star, self.network)
            t_end_enum = time.time()
            time_dict[f'enumerate_network_{split_tolerance}'] = t_end_enum - t_start_enum
            if result.result_str != "error":
                if return_tolerance:
                    return_dict['split_tolerance'] = split_tolerance
                break
        

        if result.result_str == "error":
            if return_tolerance:
                return_dict['split_tolerance'] = -1.0
            reachable_cells = set()
            if return_indices:
                reachable_cells.add((-1, -1))
            else:
                reachable_cells.add((-1, -1, -1, -1))
            
            return_dict['reachable_cells'] = reachable_cells
            if return_verts:
                return_dict['verts'] = []
            
            t_end = time.time()
            time_dict['total_time'] = t_end - t_start
            return_dict['time_dict'] = time_dict
            return return_dict 

        t_start_get_reachable = time.time()
        if return_verts:
            reachable_cells, verts = self.get_reachable_cells_from_stars(result.stars, reachable_cells, return_indices=return_indices, return_verts=True)
            return_dict['verts'] = verts
        else:
            reachable_cells = self.get_reachable_cells_from_stars(result.stars, reachable_cells, return_indices=return_indices, return_verts=False)
        t_end_get_reachable = time.time()
        time_dict['get_reachable_cells'] = t_end_get_reachable - t_start_get_reachable
        t_end = time.time()
        time_dict['total_time'] = t_end - t_start
        return_dict['time_dict'] = time_dict
        return_dict['reachable_cells'] = reachable_cells

        return return_dict