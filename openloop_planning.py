import sys
# sys.path.append('/home/khushal/PycharmProjects/Lyft-Motion-Planning/l5kit/l5kit')
sys.path.append('/Users/nicole/OSU/Lyft-Motion-Planning/l5kit/l5kit')

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import os
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

from extract_map import extract_map, always_in_bounds, get_path_cost
from path_generator import PathGenerator
import pdb

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
        x_offset = offset * np.cos(theta + np.pi/2)  # x-axis projection
        y_offset = offset * np.sin(theta + np.pi/2)  # y-axis projection
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
        
        # straight-line distance between start and goal positions
        goal_curvature = np.linalg.norm(np.array([goal_x, goal_y]) - np.array([start_x, start_y])) 

        pg = PathGenerator(start_x, start_y, start_theta, start_curvature,
                           goal_x, goal_y, goal_theta, goal_curvature,
                           alpha=10, beta=10, gamma=10, kmax=0.5)
        paths.append(pg.path)

    # test_path_x = np.round(paths[1][0])
    # test_path_y = np.round(paths[1][1])
    # print(test_path_x)
    # print(test_path_y)
    # print(lane_map)
    # for i in range(len(test_path_x)):
    #     print((test_path_x[i], test_path_y[i]))
    #     print(lane_map[(int(test_path_x[i]), int(test_path_y[i]))])

    # filter out paths that go out of bounds
    valid_paths = [p for p in paths if always_in_bounds(p, lane_map)]

    # If no valid paths just randomly pick a path so that pipeline can continue
    if len(valid_paths) == 0:
        valid_paths = [paths[np.random.choice(list(range(len(paths))))]]

    # get path costs for remaining paths 
    path_costs = [get_path_cost(p, lane_map) for p in valid_paths]

    # get path with lowest cost
    lowest_cost_path = np.argmin(path_costs)
    best_path = valid_paths[lowest_cost_path]

    # get position and yaws 
    # Path structure: [x_list, y_list, t_list, k_list]
    best_path_positions = [np.array([best_path[0][i], best_path[0][i]]) for i in range(len(best_path[0]))]
    best_path_yaws = [best_path[2][i] for i in range(len(best_path[2]))]

    return best_path_positions, best_path_yaws


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
#     position_preds_dl.append(result_dl["positions"].detach().cpu().numpy())
#     yaw_preds_dl.append(result_dl["yaws"].detach().cpu().numpy())
#
#     # Lattice planner results
#     result_lp = lattice_planner(data, rasterizer)
#     position_preds_lp.append(result_lp)
#     yaw_preds_lp.append(result_lp)
#
#     # Ground truth
#     position_gts.append(data["target_positions"].detach().cpu().numpy())
#     yaw_gts.append(data["target_yaws"].detach().cpu().numpy())
#     if idx_data == 10:
#         break

idx_data = 0
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 20):
    data = eval_dataloader.dataset[frame_number]  # ndarrays on cpu for lattice planner
    data_batch = default_collate([data])  # tensors on cuda/cpu device for deep learning model

    # Deep learning results
    result_dl = model(data_batch)
    position_preds_dl.append(result_dl["positions"].detach().cpu().numpy().squeeze())  # TODO: check shape consistencies if squeeze is required
    yaw_preds_dl.append(result_dl["yaws"].detach().cpu().numpy().squeeze())

    # Lattice planner results
    print('lattice plan for frame number {f}'.format(f=frame_number))
    result_lp_positions, result_lp_yaws = lattice_planner(data, rasterizer)  # TODO: lattice planner path
    position_preds_lp.append(result_lp_positions)  # TODO: set to list of 50 positions
    yaw_preds_lp.append(result_lp_yaws)  # TODO: set to list of 50 headings


    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    # print(result_lp_positions)
    draw_trajectory(im_ego, result_lp_positions, TARGET_POINTS_COLOR)

    plt.imshow(im_ego)
    plt.axis("off")
    plt.show()

    # Ground truth
    # position_gts.append(data["target_positions"].detach().cpu().numpy())  # TODO: check shape consistencies if squeeze is required
    # yaw_gts.append(data["target_yaws"].detach().cpu().numpy())
    position_gts.append(data["target_positions"])
    yaw_gts.append(data["target_yaws"])


    if idx_data == 10:
        break
    idx_data += 1

# position_preds_dl = np.concatenate(position_preds_dl)
# yaw_preds_dl = np.concatenate(yaw_preds_dl)
# # TODO: any required manipulation of lattice planner result arrays
# position_gts = np.concatenate(position_gts)
# yaw_gts = np.concatenate(yaw_gts)

# # Quantitative evaluation
# pos_errors = np.linalg.norm(position_preds_dl - position_gts, axis=-1)
# # TODO: get errors between LP results and GT, and LP results and DL results
# # TODO: plot new set of results below

# # DISPLACEMENT AT T
# plt.plot(np.arange(pos_errors.shape[1]), pos_errors.mean(0), label="Displacement error at T")
# plt.legend()
# plt.show()

# # ADE HIST
# plt.hist(pos_errors.mean(-1), bins=100, label="ADE Histogram")
# plt.legend()
# plt.show()

# # FDE HIST
# plt.hist(pos_errors[:,-1], bins=100, label="FDE Histogram")
# plt.legend()
# plt.show()

# angle_errors = angular_distance(yaw_preds_dl, yaw_gts).squeeze()

# # ANGLE ERROR AT T
# plt.plot(np.arange(angle_errors.shape[1]), angle_errors.mean(0), label="Angle error at T")
# plt.legend()
# plt.show()

# # ANGLE ERROR HIST
# plt.hist(angle_errors.mean(-1), bins=100, label="Angle Error Histogram")
# plt.legend()
# plt.show()

# # Qualitative evaluation
# for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 20):
#     data = eval_dataloader.dataset[frame_number]

#     data_batch = default_collate([data])

#     result_dl = model(data_batch)
#     predicted_positions = result_dl["positions"].detach().cpu().numpy().squeeze()

#     im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
#     target_positions = data["target_positions"]

#     predicted_positions = transform_points(predicted_positions, data["raster_from_agent"])
#     target_positions = transform_points(target_positions, data["raster_from_agent"])

#     draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)

#     draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)

#     plt.imshow(im_ego)
#     plt.axis("off")
#     plt.show()

# # Visualize the open-loop
# from IPython.display import display, clear_output
# import PIL

# for frame_number in range(200):
#     data = eval_dataloader.dataset[frame_number]

#     data_batch = default_collate([data])
#     data_batch = {k: v.to(device) for k, v in data_batch.items()}

#     result_dl = model(data_batch)
#     predicted_positions = result_dl["positions"].detach().cpu().numpy().squeeze()

#     predicted_positions = transform_points(predicted_positions, data["raster_from_agent"])
#     target_positions = transform_points(data["target_positions"], data["raster_from_agent"])

#     im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
#     draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
#     draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)

#     clear_output(wait=True)
#     display(PIL.Image.fromarray(im_ego))
