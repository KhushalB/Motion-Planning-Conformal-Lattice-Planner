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

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points, angular_distance
from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

import os

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
