import sys
sys.path.append('/home/khushal/PycharmProjects/Lyft-Motion-Planning/l5kit/l5kit')
# sys.path.append('/Users/nicole/OSU/Lyft-Motion-Planning/l5kit/l5kit')

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

from extract_map import extract_map
from path_generator import PathGenerator
from path_optimizer import PathOptimizer
import pdb

# Prepare data path and load cfg
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "prediction-dataset/"
# os.environ["L5KIT_DATA_FOLDER"] = '/Users/nicole/OSU/Lyft-Motion-Planning/prediction-dataset'
dm = LocalDataManager(None)
# get config
cfg = load_config_data("prediction-dataset/config.yaml")
# cfg = load_config_data("/Users/nicole/OSU/Lyft-Motion-Planning/prediction-dataset/config.yaml")

# Load the model
model_path = "prediction-dataset/planning_model_20201208.pt"
# model_path = '/Users/nicole/OSU/l5kit/pre-trained-models/planning_model_20201208.pt'
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


def plot_lattice(paths, data):
    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    for idx, path in enumerate(paths):
        positions = np.array(path[:2]).T  # shape: nx2
        positions = transform_points(positions, data["raster_from_agent"])
        if idx == 3:
            color = (0, 255, 255)
        else:
            color = (255, 0, 0)
        cv2.polylines(im_ego, np.int32([positions]), isClosed=False, color=color, thickness=1)

    plt.imshow(im_ego)
    plt.axis("off")
    plt.show()


def get_goal_states(x, y, theta, num_paths=7, offset=1.5):
    """
    Function to get list of laterally offset goal states.
    :param x: x-coordinate of goal position
    :param y: y-coordinate of goal position
    :param theta: heading of goal position (in radians in local frame)
    :param num_paths: no. of lateral goal positions to generate paths to
    :param offset: lateral offset to place goal positions; can be width of the ego vehicle
    :return: list of laterally offset goal states; each goal state is a list of the form [x, y, theta]
    """
    goal_state_set = []
    for i in range(num_paths):
        goal_offset = (i - num_paths // 2) * offset  # how much to offset goal at ith position by
        x_offset = goal_offset * np.cos(theta + np.pi/2)  # x-axis projection
        y_offset = goal_offset * np.sin(theta + np.pi/2)  # y-axis projection
        goal_state_set.append([x + x_offset, y + y_offset, theta])

    return goal_state_set


def lattice_planner(data, rasterizer):
    """
    Function to generate best trajectory using a lattice planner.
    :param data: a data sample from the dataloader
    :param rasterizer: the l5kit rasterizer
    :return: dictionary containing positions and headings along trajectory obtained using lattice planner
    """

    # get initialization parameters from the data
    lane_map, start_position, end_position, start_heading, end_heading = extract_map(data, rasterizer)
    start_x = start_position[0]
    start_y = start_position[1]
    start_theta = start_heading[0]
    start_curvature = 0
    goal_x = end_position[0]
    goal_y = end_position[1]
    goal_theta = end_heading[0]
    goal_curvature = 0

    # get list of goal states for lattice using lateral offsets
    goal_state_set = get_goal_states(goal_x, goal_y, goal_theta)

    # get optimized paths over all goal states
    paths = []
    for goal_state in goal_state_set:
        goal_x = goal_state[0]
        goal_y = goal_state[1]
        goal_theta = goal_state[2]
        # pg = PathGenerator(start_x, start_y, start_theta, start_curvature,
        #                    goal_x, goal_y, goal_theta, goal_curvature,
        #                    alpha=10, beta=10, gamma=10, kmax=0.5)
        # paths.append(pg.path)
        pg = PathOptimizer(start_x, start_y, start_theta, goal_x, goal_y, goal_theta)
        path = pg.optimize_spiral(num_samples=data['target_positions'].shape[0] + 1)  # add 1 to include start pos
        paths.append(path)

    # plot all lattice paths
    # plot_lattice(paths, data)

    path = paths[3]  # TODO: run collision checking and get path with best score
    positions = np.array(path[:2]).T  # shape: nx2
    headings = path[2][1:]  # shape: nx1; first element is start position

    return positions, headings


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
#
#     # Deep learning results
#     result_dl = model(data)
#     position_preds_dl.append(result_dl["positions"].detach().cpu().numpy())  # shape: [batch, future_num_frames, 2]
#     yaw_preds_dl.append(result_dl["yaws"].detach().cpu().numpy())  # shape: [batch, future_num_frames, 1]
#
#     # Lattice planner results
#     # postions_lp, yaws_lp = lattice_planner(data, rasterizer)
#     # position_preds_lp.append(postions_lp)
#     # yaw_preds_lp.append(yaws_lp)
#
#     # Ground truth
#     position_gts.append(data["target_positions"].detach().cpu().numpy())
#     yaw_gts.append(data["target_yaws"].detach().cpu().numpy())
#     if idx_data == 10:
#         break

idx_data = 0
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 20):
    data = eval_dataloader.dataset[frame_number]  # ndarrays on cpu for lattice planner

    # run only for samples where the vehicle is not stationary
    if ((-1e-10 < data['target_positions']) & (data['target_positions'] < 1e-10)).any():
        continue

    data_batch = default_collate([data])  # tensors on cuda/cpu device for deep learning model

    # Deep learning results
    start = time.time()
    result_dl = model(data_batch)
    stop = time.time()
    print(f'DL time: {stop-start}s')
    position_preds_dl.append(result_dl["positions"].detach().cpu().numpy())
    yaw_preds_dl.append(result_dl["yaws"].detach().cpu().numpy())

    # Lattice planner results
    start = time.time()
    print('lattice plan for frame number {f}'.format(f=frame_number))
    positions_lp, yaws_lp = lattice_planner(data, rasterizer)
    stop = time.time()
    print(f'LP time: {stop-start}s')
    position_preds_lp.append(positions_lp[np.newaxis, :])
    yaw_preds_lp.append(yaws_lp[np.newaxis, :, np.newaxis])

    # Ground truth
    position_gts.append(data["target_positions"][np.newaxis, :])
    yaw_gts.append(data["target_yaws"][np.newaxis, :])

    if idx_data == 10:
        break
    idx_data += 1

position_preds_dl = np.concatenate(position_preds_dl)
yaw_preds_dl = np.concatenate(yaw_preds_dl)
position_preds_lp = np.concatenate(position_preds_lp)
yaw_preds_lp = np.concatenate(yaw_preds_lp)
position_gts = np.concatenate(position_gts)
yaw_gts = np.concatenate(yaw_gts)

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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()

# Qualitative evaluation
LATTICE_POINTS_COLOR = (255, 0, 0)
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 20):
    data = eval_dataloader.dataset[frame_number]

    # run only for samples where the vehicle is not stationary
    if ((-1e-10 < data['target_positions']) & (data['target_positions'] < 1e-10)).any():
        continue

    data_batch = default_collate([data])

    result_dl = model(data_batch)
    predicted_positions_dl = result_dl["positions"].detach().cpu().numpy().squeeze()

    predicted_positions_lp, _ = lattice_planner(data, rasterizer)

    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    target_positions = data["target_positions"]

    predicted_positions_dl = transform_points(predicted_positions_dl, data["raster_from_agent"])
    predicted_positions_lp = transform_points(predicted_positions_lp, data["raster_from_agent"])
    target_positions = transform_points(target_positions, data["raster_from_agent"])

    draw_trajectory(im_ego, predicted_positions_dl, PREDICTED_POINTS_COLOR)
    draw_trajectory(im_ego, predicted_positions_lp, LATTICE_POINTS_COLOR)
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
    predicted_positions_dl = result_dl["positions"].detach().cpu().numpy().squeeze()

    predicted_positions_dl = transform_points(predicted_positions_dl, data["raster_from_agent"])
    target_positions = transform_points(data["target_positions"], data["raster_from_agent"])

    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
    draw_trajectory(im_ego, predicted_positions_dl, PREDICTED_POINTS_COLOR)

    clear_output(wait=True)
    display(PIL.Image.fromarray(im_ego))
