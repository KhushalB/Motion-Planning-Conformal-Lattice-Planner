import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_file_name", type=str,
                    help="config file")
file_name = parser.parse_args().config_file_name

import yaml
with open(file_name, 'r') as f:
    args = yaml.load(f)

import sys
sys.path.append(args['L5KIT_REPO_PATH'])

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points, angular_distance
from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory

from planning_utils import plot_lattice, get_goal_states, lattice_planner
from extract_map import extract_map, get_path_cost

import pdb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

TIMESTEPS = 12
USE_OLD_COST_FUNC = not args['STRAIGHT_LINE_COST_FUNC']

# Prepare data path and load cfg
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = args['L5KIT_DATA_FOLDER']
dm = LocalDataManager(None)
# get config
cfg = load_config_data(args['L5KIT_DATA_CONFIG'])

# Load the model
model_path = args['MODEL_PATH']
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
full_state_dl = []

position_preds_lp = []
yaw_preds_lp = []
full_state_lp = []

position_gts = []
yaw_gts = []
full_state_gts = []

lp_costs = []
dl_costs = []
gt_costs = []

torch.set_grad_enabled(False)


idx_data = 0
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // args['FRAME_INCREMENTS']):
    data = eval_dataloader.dataset[frame_number]  # ndarrays on cpu for lattice planner

    # run only for samples where the vehicle is not stationary
    if ((-1e-10 < data['target_positions']) & (data['target_positions'] < 1e-10)).any():
        print('skip')
        continue

    data_batch = default_collate([data])  # tensors on cuda/cpu device for deep learning model

    extract_map_outputs = extract_map(data, rasterizer)
    lane_map, start_position, end_position, start_heading, end_heading = extract_map_outputs

    # straight line path to goal (used to evaluate cost)
    straight_line_path = np.linspace(start=start_position, stop=end_position, num=TIMESTEPS)

    # Deep learning results
    start = time.time()
    result_dl = model(data_batch)
    stop = time.time()
    print(f'DL time: {stop-start}s')
    position_preds_dl.append(result_dl["positions"].detach().cpu().numpy())
    yaw_preds_dl.append(result_dl["yaws"].detach().cpu().numpy())
    state_dl = [np.array([result_dl["positions"].detach().cpu().numpy()[0][i][0], 
                result_dl["positions"].detach().cpu().numpy()[0][i][1],
                result_dl["yaws"].detach().cpu().numpy()[0][i][0]]) for i in range(TIMESTEPS)]
    full_state_dl.append(np.array([state_dl]))
    dl_cost = get_path_cost(result_dl["positions"].detach().cpu().numpy()[0], lane_map, data['raster_from_agent'], straight_line_path, dl_input=True, old_cost_func=USE_OLD_COST_FUNC)
    dl_costs.append(dl_cost)

    # Lattice planner results
    print('lattice plan for frame number {f}'.format(f=frame_number))
    start = time.time()
    positions_lp, yaws_lp, path_costs = lattice_planner(data, rasterizer, extract_map_outputs, straight_line_path, lane_checking=args['LANE_CHECKING_ON'], old_cost_func=USE_OLD_COST_FUNC)
    stop = time.time()
    print(f'LP time: {stop-start}s')
    position_preds_lp.append(positions_lp[np.newaxis, :])
    yaw_preds_lp.append(yaws_lp[np.newaxis, :, np.newaxis])
    state_lp = [np.array([positions_lp[i][0], 
                positions_lp[i][1],
                yaws_lp[i]]) for i in range(TIMESTEPS)]
    full_state_lp.append(np.array([state_lp]))
    lp_costs.append(path_costs)

    # Ground truth
    position_gts.append(data["target_positions"][np.newaxis, :])
    yaw_gts.append(data["target_yaws"][np.newaxis, :])
    state_gt = [np.array([data["target_positions"][i][0], 
                data["target_positions"][i][1],
                data["target_yaws"][i][0]]) for i in range(TIMESTEPS)]
    full_state_gts.append(np.array([state_gt]))
    gt_cost = get_path_cost(data["target_positions"][np.newaxis, :][0], lane_map, data['raster_from_agent'], straight_line_path, dl_input=True, old_cost_func=USE_OLD_COST_FUNC)
    gt_costs.append(gt_cost)

    if idx_data == args['DATA_SAMPLES']:
        break
    idx_data += 1

position_preds_dl = np.concatenate(position_preds_dl)
yaw_preds_dl = np.concatenate(yaw_preds_dl)
full_state_dl = np.concatenate(full_state_dl)
position_preds_lp = np.concatenate(position_preds_lp)
yaw_preds_lp = np.concatenate(yaw_preds_lp)
full_state_lp = np.concatenate(full_state_lp)
position_gts = np.concatenate(position_gts)
yaw_gts = np.concatenate(yaw_gts)
full_state_gts = np.concatenate(full_state_gts)

# Quantitative evaluation
pos_errors_dl = np.linalg.norm(position_preds_dl - position_gts, axis=-1)
pos_errors_lp = np.linalg.norm(position_preds_lp - position_gts, axis=-1)
num_samples = pos_errors_dl.shape[0]
future_num_frames = pos_errors_dl.shape[1]

# DISPLACEMENT AT T
# mean error at each point on path averaged over all samples; shape: [future_num_frames]
plt.title(f'Displacement at T ({num_samples} samples)')
plt.xlabel('T')
plt.ylabel('error (m)')
plt.plot(np.arange(future_num_frames), pos_errors_dl.mean(0), label="ResNet50")
plt.plot(np.arange(future_num_frames), pos_errors_lp.mean(0), label="Lattice Planner")
plt.legend()
plt.savefig(args['QUANT_SAVE_FOLDER'] + 'displacement_over_time.png')
plt.clf()

# ADE HIST
# average displacement error for the entire path for each sample; shape: [num_samples,]
ade_df = pd.DataFrame(np.concatenate((pos_errors_dl.mean(-1), pos_errors_lp.mean(-1))), columns=['ade'])
ade_df.loc[0: num_samples-1, 'model'] = 'ResNet50'
ade_df.loc[num_samples:, 'model'] = 'LatticePlanner'
ade_df['all'] = ''
ade_plot = sns.violinplot(x='ade', y='all', hue='model', data=ade_df, palette='muted', split=True, scale='count')
ade_plot.set_xlabel('ade (m)')
ade_plot.set_ylabel('')
ade_plot.set_title(f'Average Displacement Error ({num_samples} samples)')
plt.savefig(args['QUANT_SAVE_FOLDER'] + 'average_displacement_error.png')
plt.clf()

# # FDE HIST
# # final displacement error
# plt.hist(pos_errors_dl[:, -1], bins=100, label="FDE Histogram")
# plt.legend()
# plt.show()

angle_errors_dl = angular_distance(yaw_preds_dl, yaw_gts).squeeze()
angle_errors_lp = angular_distance(yaw_preds_lp, yaw_gts).squeeze()

# ANGLE ERROR AT T
# mean error at each point on path averaged over all samples; shape: [future_num_frames]
plt.title(f'Angle error at T ({num_samples} samples)')
plt.xlabel('T')
plt.ylabel('error (rad)')
plt.plot(np.arange(future_num_frames), angle_errors_dl.mean(0), label="ResNet50")
plt.plot(np.arange(future_num_frames), angle_errors_lp.mean(0), label="Lattice Planner")
plt.legend()
plt.savefig(args['QUANT_SAVE_FOLDER'] + 'angle_error_over_time.png')
plt.clf()


# ANGLE ERROR HIST
# average angle error for the entire path for each sample; shape: [num_samples,]
angle_df = pd.DataFrame(np.concatenate((angle_errors_dl.mean(-1), angle_errors_lp.mean(-1))), columns=['angle_error'])
angle_df.loc[0: num_samples-1, 'model'] = 'ResNet50'
angle_df.loc[num_samples:, 'model'] = 'LatticePlanner'
angle_df['all'] = ''
angle_plot = sns.violinplot(x='angle_error', y='all', hue='model', data=angle_df, palette='muted', split=True, scale='count')
angle_plot.set_xlabel('Angle error (rad)')
angle_plot.set_ylabel('')
angle_plot.set_title(f'Average Angle Error ({num_samples} samples)')
plt.savefig(args['QUANT_SAVE_FOLDER'] + 'average_angle_error.png')
plt.clf()

# FULL STATE ERROR AT T
state_errors_dl = np.linalg.norm(full_state_dl - full_state_gts, axis=-1)
state_errors_lp = np.linalg.norm(full_state_lp - full_state_gts, axis=-1)

# mean error at each point on path averaged over all samples; shape: [future_num_frames]
plt.title(f'State error at T ({num_samples} samples)')
plt.xlabel('T')
plt.ylabel('error')
plt.plot(np.arange(future_num_frames), state_errors_dl.mean(0), label="ResNet50")
plt.plot(np.arange(future_num_frames), state_errors_lp.mean(0), label="Lattice Planner")
plt.legend()
plt.savefig(args['QUANT_SAVE_FOLDER'] + 'state_error_over_time.png')
plt.clf()

# COST COMPARISON
plt.title(f'Cost Distribution ({num_samples} samples)')
plt.xlabel('cost')
plt.hist([dl_costs, lp_costs, gt_costs], label=['ResNet50', 'LatticePlanner', 'Ground Truth'])
plt.legend()
plt.savefig(args['QUANT_SAVE_FOLDER'] + 'cost_comparison.png')
plt.clf()


# Qualitative evaluation
idx_data = 0
LATTICE_POINTS_COLOR = (255, 0, 0)
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 12):
    data = eval_dataloader.dataset[frame_number]

    # run only for samples where the vehicle is not stationary
    if ((-1e-10 < data['target_positions']) & (data['target_positions'] < 1e-10)).any():
        continue

    data_batch = default_collate([data])

    extract_map_outputs = extract_map(data, rasterizer)
    lane_map, start_position, end_position, start_heading, end_heading = extract_map_outputs

    # straight line path to goal (used to evaluate cost)
    straight_line_path = np.linspace(start=start_position, stop=end_position, num=TIMESTEPS)

    result_dl = model(data_batch)
    predicted_positions_dl = result_dl["positions"].detach().cpu().numpy().squeeze()

    predicted_positions_lp, _, _ = lattice_planner(data, rasterizer, extract_map_outputs, straight_line_path, lane_checking=args['LANE_CHECKING_ON'], old_cost_func=USE_OLD_COST_FUNC)

    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    target_positions = data["target_positions"]

    predicted_positions_dl = transform_points(predicted_positions_dl, data["raster_from_agent"])
    predicted_positions_lp = transform_points(predicted_positions_lp, data["raster_from_agent"])
    target_positions = transform_points(target_positions, data["raster_from_agent"])

    draw_trajectory(im_ego, predicted_positions_dl, PREDICTED_POINTS_COLOR)
    draw_trajectory(im_ego, predicted_positions_lp, LATTICE_POINTS_COLOR)
    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)

    plt.imshow(im_ego)
    # plt.axis("off")
    # plt.show()
    plt.savefig(args['QUAL_SAVE_FOLDER'] + '{0}.png'.format(frame_number))
    plt.clf()

# Visualize the open-loop
# from IPython.display import display, clear_output
# import PIL

# for frame_number in range(200):
#     data = eval_dataloader.dataset[frame_number]

#     data_batch = default_collate([data])
#     data_batch = {k: v.to(device) for k, v in data_batch.items()}

#     result_dl = model(data_batch)
#     predicted_positions_dl = result_dl["positions"].detach().cpu().numpy().squeeze()

#     predicted_positions_dl = transform_points(predicted_positions_dl, data["raster_from_agent"])
#     target_positions = transform_points(data["target_positions"], data["raster_from_agent"])

#     im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
#     draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
#     draw_trajectory(im_ego, predicted_positions_dl, PREDICTED_POINTS_COLOR)

#     clear_output(wait=True)
#     display(PIL.Image.fromarray(im_ego))
