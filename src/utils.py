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
import pickle
from tqdm import tqdm

import logging

class Plotter:
    def __init__(self, p_lbs, theta_lbs, x_lims=(-11.0, 11.0), y_lims=(-30.0, 30.0)) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8,8), dpi=200)
        self.p_bounds = [np.inf, -np.inf]
        self.theta_bounds = [np.inf, -np.inf]
        self.p_lbs = p_lbs
        self.theta_lbs = theta_lbs
        self.cell_width = (p_lbs[1] - p_lbs[0])
        self.cell_height = (theta_lbs[1] - theta_lbs[0])
        self.legend_label_list = []
        self.legend_list = []
        self.ax.set_xlim(x_lims[0], x_lims[1])
        self.ax.set_ylim(y_lims[0], y_lims[1])

    def add_safety_violation_region(self, p_safety_bounds=(-10.0, 10.0), theta_safety_bounds=(-30.0, 30.0), color='red'):
        # add safety violation region
        p_axis_bounds = self.ax.get_xlim()
        theta_axis_bounds = self.ax.get_ylim()

        width = p_safety_bounds[0] - p_axis_bounds[0]
        height = theta_axis_bounds[1] - theta_axis_bounds[0]
        if width > 0 and height > 0:
            rec = plt.Rectangle((p_axis_bounds[0], theta_axis_bounds[0]), width, height, color=color, alpha=0.2)
            self.ax.add_patch(rec)

        width = -p_safety_bounds[1] + p_axis_bounds[1]
        if width > 0 and height > 0:
            rec = plt.Rectangle((p_safety_bounds[1], theta_axis_bounds[0]), width, height, color=color, alpha=0.2)
            self.ax.add_patch(rec)
        
        width = p_axis_bounds[1] - p_axis_bounds[0]
        height = theta_safety_bounds[0] - theta_axis_bounds[0]
        if width > 0 and height > 0:
            rec = plt.Rectangle((p_axis_bounds[0], theta_axis_bounds[0]), width, height, color=color, alpha=0.2)
            self.ax.add_patch(rec)
        
        height = -theta_safety_bounds[1] + theta_axis_bounds[1]
        if width > 0 and height > 0:
            rec = plt.Rectangle((p_axis_bounds[0], theta_safety_bounds[1]), width, height, color=color, alpha=0.2)
            self.ax.add_patch(rec)
        self.legend_label_list.append("Safety Violation Region")
        self.legend_list.append(rec)    

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
            cell = plt.Rectangle((x, y), self.cell_width, self.cell_height, fill=filled, linewidth=2, facecolor=color, edgecolor=color, alpha=1)
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
    
    def save_figure(self, file_name, x_range=None, y_range=None, plot_grids=False):
        if plot_grids:
            ## plot grids
            for p_lb in self.p_lbs:
                X = [p_lb, p_lb]
                Y = [self.theta_lbs[0], self.theta_lbs[-1]+self.theta_lbs[1]-self.theta_lbs[0]]
                self.ax.plot(X, Y, 'lightgray', alpha=0.2)

            for theta_lb in self.theta_lbs:
                Y = [theta_lb, theta_lb]
                X = [self.p_lbs[0], self.p_lbs[-1]+self.p_lbs[1]-self.p_lbs[0]]
                self.ax.plot(X, Y, 'lightgray', alpha=0.2)

        if x_range is not None and y_range is not None:
            self.ax.set_xlim(x_range[0], x_range[1])
            self.ax.set_ylim(y_range[0], y_range[1])
        else:    
            self.ax.set_xlim(self.p_bounds[0]-0.2, self.p_bounds[1]+0.2)
            self.ax.set_ylim(self.theta_bounds[0]-0.2, self.theta_bounds[1]+0.2)

        
        
        self.ax.set_xlabel(r"$p$ (m)")
        self.ax.set_ylabel(r"$\theta$ (degrees)")
        if len(self.legend_list) != 0:
            self.ax.legend(self.legend_list, self.legend_label_list, loc='upper right')
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

    def simulate_next_step_eagerly_searching(self, ps, thetas):
        z_samples = 500
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_name = self.session.get_outputs()[0].name
        dirs = [(1, 0), (22., 60.), (0, 1), (22, -60.)]
        p_list = []
        theta_list = []
        for idx, (p_i, theta_i) in enumerate(zip(ps, thetas)):
            zs = np.random.uniform(-0.8, 0.8, size=(z_samples, 2)).astype(np.float32)
            dir = dirs[idx % len(dirs)]
            max_dist_along_dir = 0
            p_ = None
            theta_ = None
            for z_i in zs:
                input_0 = np.concatenate([z_i, p_i, theta_i]).astype(np.float32).reshape(input_shape)
                res = self.session.run([output_name], {input_name: input_0})
                p_cdd, theta_cdd = res[0][0]
                dist_along_dir = np.abs(p_cdd * dir[0] + theta_cdd * dir[1])/np.sqrt(dir[0]**2 + dir[1]**2)
                if dist_along_dir > max_dist_along_dir:
                    max_dist_along_dir = dist_along_dir
                    p_ = p_cdd
                    theta_ = theta_cdd
            p_list.append(p_)
            theta_list.append(theta_)
        return np.array(p_list).reshape(-1, 1), np.array(theta_list).reshape(-1, 1)

class BaselineVerifier:
    def __init__(self, p_lbs, p_ubs, theta_lbs, theta_ubs, network_file_path=None, csv_file=None, p_safe_bounds=(-10.0, 10.0), theta_safe_bounds=(-30.0, 30.0)) -> None:
        self.p_lbs = p_lbs
        self.p_ubs = p_ubs
        self.theta_lbs = theta_lbs
        self.theta_ubs = theta_ubs
        self.p_safe_bounds = p_safe_bounds
        self.theta_safe_bounds = theta_safe_bounds

        assert network_file_path is not None or csv_file is not None, "Either network_file_path or csv_file should be provided"
        if csv_file is not None:
            self.control_bounds = defaultdict(tuple)
            with open(csv_file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    cell = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                    self.control_bounds[cell] = (float(row[4]), float(row[5]))
        else:
            self.network = nnenum.load_onnx_network(network_file_path)

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

        return [p_lb, p_ub], [theta_lb, theta_ub]
    
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

        assert 0<=p_lb_idx<=len(self.p_lbs), f"({p_lb_idx}, {p_ub_idx}, {theta_lb_idx}, {theta_ub_idx})"
        assert 0<=p_ub_idx<=len(self.p_ubs), f"({p_lb_idx}, {p_ub_idx}, {theta_lb_idx}, {theta_ub_idx})"
        assert 0<=theta_lb_idx<=len(self.theta_lbs), f"({p_lb_idx}, {p_ub_idx}, {theta_lb_idx}, {theta_ub_idx})"
        assert 0<=theta_ub_idx<=len(self.theta_ubs), f"({p_lb_idx}, {p_ub_idx}, {theta_lb_idx}, {theta_ub_idx})"

        for p_idx in range(p_lb_idx, p_ub_idx):
            for theta_idx in range(theta_lb_idx, theta_ub_idx):
                if return_indices:
                    reachable_cells.add((p_idx, theta_idx))
                else:
                    reachable_cells.add((self.p_lbs[p_idx], self.p_ubs[p_idx], 
                                         self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]))

        return reachable_cells
    
    def compute_next_reachable_cells(self, p_idx, theta_idx, return_indices=False, return_bounds=False):

        p_lb = self.p_lbs[p_idx]
        p_ub = self.p_ubs[p_idx]
        theta_lb = self.theta_lbs[theta_idx]
        theta_ub = self.theta_ubs[theta_idx]

        return_dict = dict()
        if hasattr(self, 'control_bounds'):
            control_bounds = self.control_bounds[(p_lb, p_ub, theta_lb, theta_ub)]
        else:
            # set nneum settings
            nnenum.set_exact_settings()
            Settings.GLPK_TIMEOUT = 10
            Settings.PRINT_OUTPUT = False
            Settings.TIMING_STATS = True
            Settings.RESULT_SAVE_STARS = True
            Settings.SPLIT_TOLERANCE = 1e-8

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

        return_dict['out_of_p_safety_bounds'] = False
        return_dict['out_of_theta_safety_bounds'] = False

        if p_bounds_[0] < self.p_safe_bounds[0] or p_bounds_[1] > self.p_safe_bounds[1]:
            return_dict['out_of_p_safety_bounds'] = True
        if theta_bounds_[0] < self.theta_safe_bounds[0] or theta_bounds_[1] > self.theta_safe_bounds[1]:
            return_dict['out_of_theta_safety_bounds'] = True
            
        # filter out the out of range cells
        p_bounds_[0] = np.clip(p_bounds_[0], self.p_lbs[0], self.p_ubs[-1])
        p_bounds_[1] = np.clip(p_bounds_[1], self.p_lbs[0], self.p_ubs[-1])
        theta_bounds_[0] = np.clip(theta_bounds_[0], self.theta_lbs[0], self.theta_ubs[-1])
        theta_bounds_[1] = np.clip(theta_bounds_[1], self.theta_lbs[0], self.theta_ubs[-1])
        
        reachable_cells = self.get_overlapping_cells_from_intervals(p_bounds_, theta_bounds_, return_indices=return_indices)
        return_dict['reachable_cells'] = reachable_cells

        if return_bounds:
            return_dict['p_bounds'] = p_bounds_
            return_dict['theta_bounds'] = theta_bounds_
        
        return return_dict

class MultiStepVerifier:
    def __init__(self, p_lbs, p_ubs, theta_lbs, theta_ubs, network_file_path=None, reachable_cells_path=None, p_safe_bounds=(-10.0, 10.0), theta_safe_bounds=(-30.0, 30.0)) -> None:

        self.p_lbs = p_lbs
        self.p_ubs = p_ubs
        self.theta_lbs = theta_lbs
        self.theta_ubs = theta_ubs
        self.p_safe_bounds = p_safe_bounds
        self.theta_safe_bounds = theta_safe_bounds
        self.p_safe_lb_idx = math.ceil((p_safe_bounds[0] - p_lbs[0])/(p_ubs[0] - p_lbs[0]))
        self.p_safe_ub_idx = math.floor((p_safe_bounds[1] - p_ubs[0])/(p_ubs[0] - p_lbs[0]))
        self.theta_safe_lb_idx = math.ceil((theta_safe_bounds[0] - theta_lbs[0])/(theta_ubs[0] - theta_lbs[0]))
        self.theta_safe_ub_idx = math.floor((theta_safe_bounds[1] - theta_ubs[0])/(theta_ubs[0] - theta_lbs[0]))
        self.latent_bounds = 0.8

        assert network_file_path is not None or reachable_cells_path is not None, "Either network_file_path or reachable_cells_path must be provided"

        if reachable_cells_path is not None:
            self.reachable_cells = pickle.load(open(reachable_cells_path, 'rb'))

        else:
            Settings.ONNX_WHITELIST.append("TaxiNetDynamics")
            shared_library = "libcustom_dynamics.so"
            so = ort.SessionOptions()
            so.register_custom_ops_library(shared_library)

            self.network = nnenum.load_onnx_network(network_file_path)
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
            #raise Exception("p out of range")
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
        lp_false = False
        for star in stars:

            # compute the interval enclosure for the star set to get the candidate cells
            ## TODO: using zonotope enclosure may be faster,
            ## but we need to solve LPs to get the candidate cells
            try:
                interval_enclosure = self.compute_interval_enclosure(star)
            except:
                logging.warning(f"    warning: error in computing interval enclosure for star, skip for now")
                lp_false = True
                break

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
                    # sanity check if reachable cells from degraded method is provided
                    if self.reachable_cells_from_degraded_method is not None and self.reachable_cells_from_degraded_method != {(-2, -2)}:
                        if (interval_enclosure[0][0], interval_enclosure[1][0]) not in self.reachable_cells_from_degraded_method:
                            logging.warning(f"    warning: error in computing reachable cells for p_idx={p_idx}, theta_idx={theta_idx}, this should be overapproximated by a degraded method")
                else:
                    reachable_cells.add((self.p_lbs[interval_enclosure[0][0]], self.p_ubs[interval_enclosure[0][0]],
                                         self.theta_lbs[interval_enclosure[1][0]], self.theta_ubs[interval_enclosure[1][0]]))
                continue

            # intersection check for the candidate cells
            for p_idx in range(interval_enclosure[0][0], interval_enclosure[0][1]):
                for theta_idx in range(interval_enclosure[1][0], interval_enclosure[1][1]):
                    if self.reachable_cells_from_degraded_method is not None and self.reachable_cells_from_degraded_method != {(-2, -2)} and (p_idx, theta_idx) not in self.reachable_cells_from_degraded_method:
                            continue

                    if return_indices and (p_idx, theta_idx) not in reachable_cells:
                        try:
                            if self.check_intersection(star, p_idx, theta_idx):
                                reachable_cells.add((p_idx, theta_idx))
                        except:
                            logging.warning(f"    warning: error in checking intersection for p_idx={p_idx}, theta_idx={theta_idx}")
                    elif (not return_indices) and (self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]) not in reachable_cells:
                        if self.check_intersection(star, p_idx, theta_idx):
                            reachable_cells.add(((self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx])))

        if return_verts:
            return reachable_cells, verts
        else:
            return reachable_cells, lp_false


    def compute_next_reachable_cells(self, p_idx, theta_idx, reachable_cells_from_degraded_method=None, return_indices=False, return_verts=False, print_output=False, pbar=None, return_tolerance=False, single_thread=False, start_tol=1e-8, end_tol=1e-2):
        p_lb = self.p_lbs[p_idx]
        p_ub = self.p_ubs[p_idx]
        theta_lb = self.theta_lbs[theta_idx]
        theta_ub = self.theta_ubs[theta_idx]
        self.reachable_cells_from_degraded_method = reachable_cells_from_degraded_method
        assert self.reachable_cells_from_degraded_method is not None, "reachable_cells_from_degraded_method must be provided"

        return_dict = dict()
        if hasattr(self, 'reachable_cells'):
            assert return_indices
            assert not return_verts

            return_dict["out_of_p_safety_bounds"] = False
            return_dict["out_of_theta_safety_bounds"] = False

            if (p_idx, theta_idx) in self.reachable_cells:
                if self.reachable_cells[(p_idx, theta_idx)] == {(-1, -1)}:
                    logging.info(f"warning: error in computing reachable cells for p_idx={p_idx}, theta_idx={theta_idx}, this should be overapproximated by a degraded method")
                    return_dict["reachable_cells"] = set()
                elif self.reachable_cells[(p_idx, theta_idx)] == {(-2, -2)}:
                    return_dict["reachable_cells"] = self.reachable_cells[(p_idx, theta_idx)] 
                    return_dict["out_of_p_safety_bounds"] = True
                else:
                    reachable_cells = set()
                    for reachable_cell in self.reachable_cells[(p_idx, theta_idx)]:
                        if reachable_cell == (-3, -3):
                            reachable_cells.add((-3, -3))
                            return_dict["out_of_theta_safety_bounds"] = True
                            continue

                        elif reachable_cell[0] > self.p_safe_ub_idx or reachable_cell[0] < self.p_safe_lb_idx:
                            reachable_cells = {(-2, -2)}
                            return_dict["out_of_p_safety_bounds"] = True
                            break

                        elif reachable_cell[1] > self.theta_safe_ub_idx or reachable_cell[1] < self.theta_safe_lb_idx:
                            return_dict["out_of_theta_safety_bounds"] = True
                            reachable_cells.add((-3, -3))
                        
                        else:
                            reachable_cells.add(reachable_cell)
                    
                    return_dict["reachable_cells"] = reachable_cells

        else:
            reachable_cells = set()
            time_dict = defaultdict()
            t_start = time.time()



            # simulations
            t_start_sim = time.time()
            samples = 10000
            z = np.random.uniform(-self.latent_bounds, self.latent_bounds, size=(samples, self.step*2)).astype(np.float32)
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

                # if p is out of the range (unsafe), then only return (-2, -2), break
                if p_idx < 0 or p_idx >= len(self.p_lbs):
                    reachable_cells = set()
                    if return_indices:
                        reachable_cells.add((-2, -2))
                    else:
                        reachable_cells.add((-2, -2, -2, -2))
                    break

                # if theta is out of the range, then return (-3, -3), but continue
                if theta_idx < 0 or theta_idx >= len(self.theta_lbs):
                    if return_indices:
                        reachable_cells.add((-3, -3))
                        if self.reachable_cells_from_degraded_method is not None and reachable_cells_from_degraded_method != {(-2, -2)}:
                            if (-3, -3) not in reachable_cells_from_degraded_method:
                                logging.warning(f"    warning: error in computing reachable cells for p_idx={p_idx}, theta_idx={theta_idx}, this should be overapproximated by a degraded method")
                    else:
                        reachable_cells.add((-3, -3, -3, -3))
                    continue
                
                # normal case
                if return_indices:
                    reachable_cells.add((p_idx, theta_idx))
                    # sanity check if reachable cells from degraded method is provided
                    if self.reachable_cells_from_degraded_method is not None and reachable_cells_from_degraded_method != {(-2, -2)}:
                        if (p_idx, theta_idx) not in reachable_cells_from_degraded_method:
                            logging.warning(f"    warning: error in computing reachable cells for p_idx={p_idx}, theta_idx={theta_idx}, this should be overapproximated by a degraded method")
                else:
                    reachable_cells.add((self.p_lbs[p_idx], self.p_ubs[p_idx], 
                                         self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]))

            t_end_sim = time.time()
            time_dict['simulation'] = t_end_sim - t_start_sim

            return_dict = dict()
            return_dict['simulation_reachable_cells'] = reachable_cells

            if reachable_cells == {(-2, -2)} or reachable_cells == {(-2, -2, -2, -2)}:
                # in this case, the airplane is out of the safe region
                logging.info(f"    Simulation done, found unsafe region.")
                # prepare the return
                return_dict['reachable_cells'] = reachable_cells
                if return_verts:
                    return_dict['verts'] = []
                if return_tolerance:
                    return_dict['split_tolerance'] = -2.0 # this indicates that only simulation is used
                time_dict['total_time'] = time_dict['simulation']
                return_dict['time_dict'] = time_dict
                return return_dict
            
            logging.info(f"    Simulation done, found {len(reachable_cells)} reachable cells.")

            # set nneum settings
            nnenum.set_exact_settings()
            Settings.GLPK_TIMEOUT = 10
            Settings.PRINT_OUTPUT = print_output
            Settings.TIMING_STATS = False
            #Settings.RESULT_SAVE_STARS = True
            #Settings.CONTRACT_LP_OPTIMIZED = False # use optimized lp contraction
            Settings.CONTRACT_LP_TRACK_WITNESSES = False
            #Settings.CONTRACT_LP_CHECK_EPSILON = 1e-3 # numerical error tolerated when doing contractions before error, None=skip

            if single_thread:
                Settings.NUM_PROCESSES = 1


            init_box = [[-self.latent_bounds, self.latent_bounds], [-self.latent_bounds, self.latent_bounds]]
            init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
            init_box.extend([[-self.latent_bounds, self.latent_bounds]]*((self.step-1)*2))
            init_box = np.array(init_box, dtype=np.float32)
            init_bm, init_bias, init_box = compress_init_box(init_box)
            star = LpStar(init_bm, init_bias, init_box)

            info_send_to_nnumem = dict()
            info_send_to_nnumem['p_lbs'] = self.p_lbs
            info_send_to_nnumem['p_ubs'] = self.p_ubs
            info_send_to_nnumem['theta_lbs'] = self.theta_lbs
            info_send_to_nnumem['theta_ubs'] = self.theta_ubs
            possible_cells = self.reachable_cells_from_degraded_method-reachable_cells
            assert reachable_cells.issubset(self.reachable_cells_from_degraded_method)
            info_send_to_nnumem['possible_cells'] = possible_cells
            
            split_tolerance = start_tol
            while split_tolerance <= end_tol:
            #for split_tolerance in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                t_start_enum = time.time()
                logging.info(f"    Start enumerating with split_tolerance={split_tolerance}")

                Settings.SPLIT_TOLERANCE = split_tolerance # small outputs get rounded to zero when deciding if splitting is possible
                result = nnenum.enumerate_network(star, self.network, info_send_to_nnumem)
                t_end_enum = time.time()
                time_dict[f'enumerate_network_{split_tolerance}'] = t_end_enum - t_start_enum
                if result.result_str != "error":
                    logging.info(f"    Enumerating done, found {len(result.reachable_cells)} new reachable_cells.")
                    if return_tolerance:
                        return_dict['split_tolerance'] = split_tolerance
                    break
                split_tolerance *= 10
        

            if result.result_str == "error":
                logging.warning(f"    Enumerating failed with all split_tolerance.")
                if self.reachable_cells_from_degraded_method is not None:
                    reachable_cells = self.reachable_cells_from_degraded_method
                    logging.info(f"    Use degraded method to overapproximate reachable cells, found {len(reachable_cells)} reachable cells.")
                    return_dict['split_tolerance'] = -3.0
                else:
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
            reachable_cells = result.reachable_cells | reachable_cells
            #reachable_cells, lp_false = self.get_reachable_cells_from_stars(result.stars, reachable_cells, return_indices=return_indices, return_verts=False)
            t_end_get_reachable = time.time()
            error_stars = result.total_error_stars

            time_dict['get_reachable_cells'] = t_end_get_reachable - t_start_get_reachable
            t_end = time.time()
            time_dict['total_time'] = t_end - t_start
            return_dict['time_dict'] = time_dict
            return_dict['reachable_cells'] = reachable_cells
            return_dict['error_stars'] = error_stars

        return return_dict

def compute_unsafe_cells(reachable_sets, p_lbs, p_ubs, theta_lbs, theta_ubs, p_safe_bounds=(-10.0, 10.0)):
    isSafe = np.ones((len(p_lbs), len(theta_lbs)))
    reversed_reachable_sets = defaultdict(set)
    helper = set()
    new_unsafe_state = []

    for p_idx in tqdm(range(len(p_lbs))):
        for theta_idx in tqdm(range(len(theta_lbs)), leave=False):
            
            if reachable_sets[(p_idx, theta_idx)] == {(-1, -1)} or p_lbs[p_idx] < p_safe_bounds[0] or p_ubs[p_idx] > p_safe_bounds[1]:
                isSafe[p_idx, theta_idx] = 0
                helper.add((p_idx, theta_idx))
                new_unsafe_state.append((p_idx, theta_idx))
                continue
                
            for reachable_cell in reachable_sets[(p_idx, theta_idx)]:
                if reachable_cell == (-2, -2):
                    isSafe[p_idx, theta_idx] = 0
                    helper.add((p_idx, theta_idx))
                    new_unsafe_state.append((p_idx, theta_idx))
                    break
                if reachable_cell == (-3, -3) or reachable_cell[1] > len(theta_lbs)-1:
                    continue

                assert len(p_lbs)>=reachable_cell[0] >= 0, f"reachable_cell: {reachable_cell}"
                assert len(theta_lbs)>=reachable_cell[1] >= 0, f"reachable_cell: {reachable_cell}"
                reversed_reachable_sets[reachable_cell].add((p_idx, theta_idx))
    
    while len(new_unsafe_state)>0:
        temp = []
        for i, j in new_unsafe_state:
            for (_i, _j) in reversed_reachable_sets[(i, j)]:
                if (_i, _j) not in helper:
                    isSafe[_i, _j] = 0
                    temp.append((_i, _j))
                    helper.add((_i, _j))
        new_unsafe_state = temp
    
    return isSafe
