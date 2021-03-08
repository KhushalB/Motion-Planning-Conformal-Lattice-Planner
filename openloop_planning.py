import sys
# sys.path.append('/home/khushal/PycharmProjects/Lyft-Motion-Planning/l5kit/l5kit')
sys.path.append('/Users/nicole/OSU/Lyft-Motion-Planning/l5kit/l5kit')

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points, angular_distance
from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

from extract_map import extract_map
from path_generator import PathGenerator
import os

# Prepare data path and load cfg
# set env variable for data
# os.environ["L5KIT_DATA_FOLDER"] = "prediction-dataset/"
os.environ["L5KIT_DATA_FOLDER"] = '/Users/nicole/OSU/Lyft-Motion-Planning/prediction-dataset'
dm = LocalDataManager(None)
# get config
# cfg = load_config_data("prediction-dataset/config.yaml")
cfg = load_config_data("/Users/nicole/OSU/Lyft-Motion-Planning/prediction-dataset/config.yaml")

# Load the model
# model_path = "prediction-dataset/planning_model_20201208.pt"
model_path = '/Users/nicole/OSU/l5kit/pre-trained-models/planning_model_20201208.pt'
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

# for idx_data, data_ in enumerate(tqdm(eval_dataloader)):
#     data = {k: v.to(device) for k, v in data_.items()}
#     result_dl = model(data)
#     position_preds_dl.append(result_dl["positions"].detach().cpu().numpy())
#     yaw_preds_dl.append(result_dl["yaws"].detach().cpu().numpy())
idx_data = 0
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 20):
    data = eval_dataloader.dataset[frame_number]

    data_batch = default_collate([data])

    result_dl = model(data_batch)
    predicted_positions = result_dl["positions"].detach().cpu().numpy().squeeze()
    print('dl predicted positions')
    print(predicted_positions)
    yaw_positions = result_dl["yaws"].detach().cpu().numpy().squeeze()
    
    position_preds_dl.append(predicted_positions)
    yaw_preds_dl.append(yaw_positions)

    # TODO: Lattice Planner results
    print('lattice plan for frame number {f}'.format(f=frame_number))
    our_map, start_position, end_position, start_heading, end_heading = extract_map(data, rasterizer)
    start_x = start_position[0]
    start_y = start_position[1]
    start_theta = start_heading
    start_curvature = 0
    goal_x = end_position[0]
    goal_y = end_position[1]
    goal_theta = end_heading
    goal_curvature = 1
    pg = PathGenerator(start_x, start_y, start_theta, start_curvature, 
        goal_x, goal_y, goal_theta, goal_curvature,
        alpha=10, beta=10, gamma=10, kmax=0.5)
    print(pg.path)
    # position_preds_lp.append(result_lp["positions"].detach().cpu().numpy())
    # yaw_preds_lp.append(result_lp["yaws"].detach().cpu().numpy())

    # position_gts.append(data["target_positions"].detach().cpu().numpy())
    # yaw_gts.append(data["target_yaws"].detach().cpu().numpy())
    position_gts.append(data["target_positions"])
    yaw_gts.append(data["target_yaws"])
    
    if idx_data == 10:
        break
    idx_data += 1

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
