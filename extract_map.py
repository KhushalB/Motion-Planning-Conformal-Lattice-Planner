from l5kit.geometry import transform_points
import matplotlib.pyplot as plt
import numpy as np


ROAD_COLOR = np.array([17, 17, 31])
EGO_VEHICLE = np.array([0, 255, 0])
OTHER_VEHICLE = np.array([0, 0, 255])
OFF_ROAD = np.array([255, 255, 255])


def extract_map(data, rasterizer):
    im = data["image"].transpose(1, 2, 0)
    im = rasterizer.to_rgb(im)
    history_positions_pixels = transform_points(data["history_positions"],
                                                data["raster_from_agent"])
    target_positions_pixels = transform_points(data["target_positions"],
                                               data["raster_from_agent"])

    num_col = im.shape[0]
    num_row = im.shape[1]
    our_map = np.zeros((num_col, num_row))

    for col in range(num_col):
        for row in range(num_row):
            cell = im[col][row]
            if (np.all((ROAD_COLOR - 10) <= cell) and np.all((ROAD_COLOR + 10) >= cell)) or \
                (np.all((EGO_VEHICLE - 10) <= cell) and np.all((EGO_VEHICLE + 10) >= cell)):
                our_map[col, row] = 1
            # if np.array_equal(cell, ROAD_COLOR) or \
            #         np.array_equal(cell, EGO_VEHICLE):
            #    our_map[col, row] = 1
            elif (np.all((OTHER_VEHICLE - 10) <= cell) and np.all((OTHER_VEHICLE + 10) >= cell)):
                our_map[col, row] = 3
            else:
                # if not np.array_equal(cell, OFF_ROAD):
                if not (np.all((OFF_ROAD - 10) <= cell)):
                    our_map[col, row] = 2

    # start_position = history_positions_pixels[0]
    # end_position = target_positions_pixels[-1]
    start_position = data["history_positions"][0]
    end_position = data["target_positions"][-1]

    start_heading = data['history_yaws'][0]
    end_heading = data['target_yaws'][-1]

    return our_map, start_position, end_position, start_heading, end_heading


# Path structure: [x_list, y_list, t_list, k_list]
def always_in_bounds(path, our_map, data_raster, strict=True):
    x_list = path[0]
    y_list = path[1]

    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        tp = transform_points(np.array([np.array([x, y])]), data_raster)
        x = int(np.round(tp[0][0]))
        y = int(np.round(tp[0][1]))

        if strict:
            if our_map[(y, x)] != 1:
                # print('offroad')
                # print(y, x)
                return False

        if (our_map[(y, x)] == 0) or (our_map[(y, x)] == 3):
            return False
    return True


def always_in_bounds_no_collision(path, our_map):
    pass

def dist_from_lane_bound(path_coord, our_map):
    map_bounds = our_map.shape
    COL = 0
    ROW = 1
    num_col = map_bounds[COL]
    num_row = map_bounds[ROW]

    dist_to_north_bound = 0
    current_coord = path_coord
    while our_map[current_coord] != 2:
        if (current_coord[ROW] - 1 < 0) or (dist_to_north_bound > 10):
            break
        dist_to_north_bound += 1
        current_coord = (current_coord[COL], current_coord[ROW] - 1)

    dist_to_south_bound = 0
    current_coord = path_coord
    while our_map[current_coord] != 2:
        if (current_coord[ROW] + 1 >= num_row) or (dist_to_south_bound > 10):
            break
        dist_to_south_bound += 1
        current_coord = (current_coord[COL], current_coord[ROW] + 1)

    dist_to_east_bound = 0
    current_coord = path_coord
    while our_map[current_coord] != 2:
        if (current_coord[COL] + 1 >= num_col) or (dist_to_east_bound > 10):
            break
        dist_to_east_bound += 1
        current_coord = (current_coord[COL] + 1, current_coord[ROW])

    dist_to_west_bound = 0
    current_coord = path_coord
    while our_map[current_coord] != 2:
        if (current_coord[COL] - 1 < 0) or (dist_to_west_bound > 10):
            break
        dist_to_west_bound += 1
        current_coord = (current_coord[COL] - 1, current_coord[ROW])

    return np.min([dist_to_north_bound, dist_to_west_bound, 
        dist_to_east_bound, dist_to_south_bound])


# Path structure: [x_list, y_list, t_list, k_list]
def get_path_cost(path, our_map, data_raster, straight_line_path, dl_input=False, old_cost_func=False):
    count_cross_lane_boundary = 0
    min_dist_from_lane_bound = []

    straight_line_path = transform_points(straight_line_path, data_raster)
    path_coords = []

    if not dl_input:
        x_list = path[0]
        y_list = path[1]
    else:
        path = transform_points(path, data_raster)
        x_list = path

    for i in range(len(x_list)):
        if not dl_input:
            x = x_list[i]
            y = y_list[i]
            tp = transform_points(np.array([np.array([x, y])]), data_raster)
            x = int(np.round(tp[0][0]))
            y = int(np.round(tp[0][1]))
        else:
            x = int(np.round(path[i][0]))
            y = int(np.round(path[i][1]))

        path_coords.append([x, y])

        if old_cost_func:
            # if crossing a lane boundary
            if our_map[(y, x)] == 2:
                count_cross_lane_boundary += 1
            
            # if not crossing a lane boundary, get minimum distance from lane boundary
            else:
                d = dist_from_lane_bound((y, x), our_map)
                min_dist_from_lane_bound.append(d)

    straight_line_path_cost = np.linalg.norm(straight_line_path - np.array(path_coords))
    
    if old_cost_func:
        cost = count_cross_lane_boundary - np.mean(min_dist_from_lane_bound)
    else:
        cost = straight_line_path_cost
    # print("path cost {c}".format(c=cost))
    return cost
