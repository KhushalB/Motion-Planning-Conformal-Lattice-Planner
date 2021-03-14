import numpy as np

from extract_map import always_in_bounds, extract_map, get_path_cost
from path_generator import PathGenerator
from path_optimizer import PathOptimizer

def plot_lattice(paths, data, rasterizer):
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


def lattice_planner(data, rasterizer, extract_map_outputs, straight_line_path, lane_checking=True, old_cost_func=False):
    """
    Function to generate best trajectory using a lattice planner.
    :param data: a data sample from the dataloader
    :param rasterizer: the l5kit rasterizer
    :return: dictionary containing positions and headings along trajectory obtained using lattice planner
    """

    # get initialization parameters from the data
    lane_map, start_position, end_position, start_heading, end_heading = extract_map_outputs
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

    if lane_checking:
        # filter out paths that go out of bounds or drive over lanes
        valid_paths = [p for p in paths if always_in_bounds(p, lane_map, data['raster_from_agent'])]

        path_cost = 0
        if len(valid_paths) == 0:
            # filter out paths that go out of bounds, but still drive over lanes
            valid_paths = [p for p in paths if always_in_bounds(p, lane_map, data['raster_from_agent'], strict=False)]

            # If no valid paths just pick middle path
            if len(valid_paths) == 0:
                path = paths[3]
                path_cost = get_path_cost(paths[3], lane_map, data['raster_from_agent'], straight_line_path, old_cost_func=old_cost_func)
            else:
                # print('found valid paths')
                # plot_lattice(valid_paths, data)

                # get path costs for remaining paths 
                path_costs = [get_path_cost(p, lane_map, data['raster_from_agent'], straight_line_path, old_cost_func=old_cost_func) for p in valid_paths]

                # get path with lowest cost
                lowest_cost_path = np.argmin(path_costs)
                best_path = valid_paths[lowest_cost_path]
                
                path = best_path
                path_cost = path_costs[lowest_cost_path]
        else:
            # get path costs for remaining paths 
            path_costs = [get_path_cost(p, lane_map, data['raster_from_agent'], straight_line_path, old_cost_func=old_cost_func) for p in valid_paths]

            # get path with lowest cost
            lowest_cost_path = np.argmin(path_costs)
            best_path = valid_paths[lowest_cost_path]
            
            path = best_path
            path_cost = path_costs[lowest_cost_path]

    else:
        path = paths[3]
        path_cost = get_path_cost(path, lane_map, data['raster_from_agent'], straight_line_path, old_cost_func=old_cost_func)

    # plot_lattice([path], data)
    
    positions = np.array(path[:2]).T  # shape: nx2
    headings = path[2][1:]  # shape: nx1; first element is start position

    return positions, headings, path_cost
