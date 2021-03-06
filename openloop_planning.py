import sys
sys.path.append('/home/khushal/PycharmProjects/Lyft-Motion-Planning/l5kit/l5kit')

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from math import sin, cos
from scipy.integrate import quad
from scipy.optimize import minimize

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points, angular_distance
from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

import os


def lattice_planner(x0, y0, heading_i, curvature_i, xf, yf, heading_f, curvature_f):
    t0, k0 = heading_i, curvature_i
    tf, kf = heading_f, curvature_f

    path = "Path that satisfies kinematic constraints"

    # NOTE: at this point we don't know sf
    sf = 3
    p1 = 1
    p2 = 1

    p0 = k0
    #p1 = k_s(sf/3, t_params)
    #p2 = k_s(2*sf/3, t_params)
    p3 = kf
    p4 = sf

    # NOTE: the parameters we are solving are p1, p2, and p4

    a0_p = remap_a0(p0)
    a1_p = remap_a1(p0,p1,p2,p3,p4)
    a2_p = remap_a2(p0,p1,p2,p3,p4)
    a3_p = remap_a3(p0,p1,p2,p3,p4)

    t_params = [a0_p, a1_p, a2_p, a3_p, t0]
    
    # Tunable variables
    k_max = 0.5
    alpha = 10
    beta = 10
    gamma = 10

    f_be = obj_function(a0_p, a1_p, a2_p, a3_p, p4)
    x_sc = x_soft(alpha, p4, xf, x0, t_params)
    y_sc = y_soft(beta, p4, yf, y0, t_params)
    t_sc = theta_soft(gamma, p4, tf, t_params)
    # This is our overall objective function below
    # Optimization parameters are p1, p2, and p4
    min_obj = f_be + x_sc + y_sc + t_sc
    
    constraint_1 = abs(p1) < k_max
    constraint_2 = abs(p2) < k_max

    guess = 3
    # result = minimize(min_obj, guess, method='L-BFGS-B',
    #                             jac=objective_jacobian, bounds=bounds,
    #                             options={'disp': True})

    return path

def k_s(s, t_params):
    """ Our cubic spiral equation """
    a0,a1,a2,a3 = tp[1],tp[2],tp[3],tp[4]
    return a3*s**3 + a2*s**2 + a1*s + a0

def obj_integrand(s, a0, a1, a2, a3):
    """ Integrand to use with objective_function() """
    return (a3*s**3 + a2*s**2 + a1*s + a0)**2

def obj_function(a0, a1, a2, a3, sf):
    """ Objective function, using the quad integral solver
    from SciPy on our objective_integrand (variable 's') 
    from 0 to sf, using coefficients a0, a1, a2, and a3 """
    return quad(obj_integrand, 0, sf, args=(a0,a1,a2,a3))

def x_soft(alpha, p4, xf, x0, theta_params):
    """ Soft inequality constraints, allows a small
    margin of error between goal point and final point
    in the curve. Scaled by alpha. """
    return alpha*(x_s(p4, x0, theta_params)-xf)

def y_soft(beta, p4, yf, y0, theta_params):
    """ Soft inequality constraints, allows a small
    margin of error between goal point and final point
    in the curve. Scaled by beta. """
    return beta*(y_s(p4, y0, theta_params)-yf)

def theta_soft(gamma, p4, tf, theta_params):
    """ Soft inequality constraints, allows a small
    margin of error between goal point and final point
    in the curve. Scaled by gamma. """
    return gamma*(theta_s(p4, theta_params)-tf)

def x_s(s, x0, theta_params):
    """ Estimates x value at location 's' along curve. Requires
    starting x value, as well as args to find theta(s). Uses
    Simpson's rule to divide domain into n=8 sections. """
    n0 = cos(theta_s(0, theta_params))
    n1 = 4*cos(theta_s(1*s/8, theta_params))
    n2 = 2*cos(theta_s(2*s/8, theta_params))
    n3 = 4*cos(theta_s(3*s/8, theta_params))
    n4 = 2*cos(theta_s(4*s/8, theta_params))
    n5 = 4*cos(theta_s(5*s/8, theta_params))
    n6 = 2*cos(theta_s(6*s/8, theta_params))
    n7 = 4*cos(theta_s(7*s/8, theta_params))
    n8 = cos(theta_s(s, theta_params))
    n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
    return x0 + (s/24)*(n_sum)

def y_s(s, y0, theta_params):
    """ Estimates y value at location 's' along curve. Requires
    starting y value, as well as args to find theta(s). Uses
    Simpson's rule to divide domain into n=8 sections. """
    n0 = sin(theta_s(0, theta_params))
    n1 = 4*sin(theta_s(1*s/8, theta_params))
    n2 = 2*sin(theta_s(2*s/8, theta_params))
    n3 = 4*sin(theta_s(3*s/8, theta_params))
    n4 = 2*sin(theta_s(4*s/8, theta_params))
    n5 = 4*sin(theta_s(5*s/8, theta_params))
    n6 = 2*sin(theta_s(6*s/8, theta_params))
    n7 = 4*sin(theta_s(7*s/8, theta_params))
    n8 = sin(theta_s(s, theta_params))
    n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
    return x0 + (s/24)*(n_sum)

def theta_s(s, tp):
    """ Finds theta value at location 's' along curve.
    Takes in theta parameters 'tp', which are the initial theta
    value and the curve's polynomial coefficients a0-a3 """
    t0,a0,a1,a2,a3 = tp[0],tp[1],tp[2],tp[3],tp[4]
    s4 = a3 * s**4 / 4
    s3 = a2 * s**3 / 3
    s2 = a1 * s**2 / 2
    s1 = a0 * s
    return t0+s4+s3+s2+s1

def remap_a0(p0):
    """ Map optimization params back to
    spiral coefficients. """
    return p0

def remap_a1(p0, p1, p2, p3, p4):
    """ Map optimization params back to
    spiral coefficients. """
    num = -1*(11*p0/2 - 9*p1 + 9*p2/2 - p3)
    denom = p4
    return num/denom

def remap_a2(p0, p1, p2, p3, p4):
    """ Map optimization params back to
    spiral coefficients. """
    num = 9*p0 - 45*p1/2 + 18*p2 - 9*p3/2
    denom = p4**2
    return num/denom

def remap_a3(p0, p1, p2, p3, p4):
    """ Map optimization params back to
    spiral coefficients. """
    num = -1*(9*p0/2 - 27*p1/2 + 27*p2/2 - 9*p3/2)
    denom = p4**3
    return num/denom

# Prepare data path and load cfg
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "prediction-dataset/"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("prediction-dataset/config.yaml")

# Load the model
model_path = "prediction-dataset/planning_model_20201208.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path).to(device)
model = model.eval()

# Load the evaluation data
eval_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"],
                             num_workers=eval_cfg["num_workers"])
print(eval_dataset)

# Evaluation loop
position_preds_dl = []
yaw_preds_dl = []

position_preds_lp = []
yaw_preds_lp = []

position_gts = []
yaw_gts = []

torch.set_grad_enabled(False)

for idx_data, data in enumerate(tqdm(eval_dataloader)):
    data = {k: v.to(device) for k, v in data.items()}
    result_dl = model(data)
    position_preds_dl.append(result_dl["positions"].detach().cpu().numpy())
    yaw_preds_dl.append(result_dl["yaws"].detach().cpu().numpy())

    # TODO: Lattice Planner results
    our_map, start_position, end_position, start_heading, end_heading = extract_map(data, rasterizer)
    # result_lp = lattice_planner()
    # position_preds_lp.append(result_lp["positions"].detach().cpu().numpy())
    # yaw_preds_lp.append(result_lp["yaws"].detach().cpu().numpy())

    position_gts.append(data["target_positions"].detach().cpu().numpy())
    yaw_gts.append(data["target_yaws"].detach().cpu().numpy())
    if idx_data == 10:
        break

position_preds_dl = np.concatenate(position_preds_dl)
yaw_preds_dl = np.concatenate(yaw_preds_dl)
# TODO: any required manipulation of lattice planner result arrays
position_gts = np.concatenate(position_gts)
yaw_gts = np.concatenate(yaw_gts)

# Quantitative evaluation
pos_errors = np.linalg.norm(position_preds_dl - position_gts, axis=-1)
# TODO: get errors between LP results and GT, and LP results and DL results
# TODO: plot new set of results below

# DISPLACEMENT AT T
plt.plot(np.arange(pos_errors.shape[1]), pos_errors.mean(0), label="Displacement error at T")
plt.legend()
plt.show()

# ADE HIST
plt.hist(pos_errors.mean(-1), bins=100, label="ADE Histogram")
plt.legend()
plt.show()

# FDE HIST
plt.hist(pos_errors[:,-1], bins=100, label="FDE Histogram")
plt.legend()
plt.show()

angle_errors = angular_distance(yaw_preds_dl, yaw_gts).squeeze()

# ANGLE ERROR AT T
plt.plot(np.arange(angle_errors.shape[1]), angle_errors.mean(0), label="Angle error at T")
plt.legend()
plt.show()

# ANGLE ERROR HIST
plt.hist(angle_errors.mean(-1), bins=100, label="Angle Error Histogram")
plt.legend()
plt.show()

# Qualitative evaluation
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 20):
    data = eval_dataloader.dataset[frame_number]

    data_batch = default_collate([data])

    result_dl = model(data_batch)
    predicted_positions = result_dl["positions"].detach().cpu().numpy().squeeze()

    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    target_positions = data["target_positions"]

    predicted_positions = transform_points(predicted_positions, data["raster_from_agent"])
    target_positions = transform_points(target_positions, data["raster_from_agent"])

    draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)

    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)

    plt.imshow(im_ego)
    plt.axis("off")
    plt.show()

# Visualize the open-loop
from IPython.display import display, clear_output
import PIL

for frame_number in range(200):
    data = eval_dataloader.dataset[frame_number]

    data_batch = default_collate([data])
    data_batch = {k: v.to(device) for k, v in data_batch.items()}

    result_dl = model(data_batch)
    predicted_positions = result_dl["positions"].detach().cpu().numpy().squeeze()

    predicted_positions = transform_points(predicted_positions, data["raster_from_agent"])
    target_positions = transform_points(data["target_positions"], data["raster_from_agent"])

    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
    draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)

    clear_output(wait=True)
    display(PIL.Image.fromarray(im_ego))
