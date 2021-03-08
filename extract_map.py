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
            if np.array_equal(cell, ROAD_COLOR) or \
                    np.array_equal(cell, EGO_VEHICLE):
                our_map[col, row] = 1
            elif np.array_equal(cell, OTHER_VEHICLE):
                our_map[col, row] = 3
            else:
                if not np.array_equal(cell, OFF_ROAD):
                    our_map[col, row] = 2

    start_position = history_positions_pixels[0]
    end_position = target_positions_pixels[-1]

    start_heading = data['history_yaws'][0]
    end_heading = data['target_yaws'][-1]

    return our_map, start_position, end_position, start_heading, end_heading


def always_in_bounds(path, our_map):
    for i, p in enumerate(path):
        x = round(p[0])
        y = round(p[1])

        if our_map[(x, y)] == 0:
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
        if current_coord[ROW] - 1 < 0:
            break
        dist_to_north_bound += 1
        current_coord = (current_coord[COL], current_coord[ROW] - 1)

    dist_to_south_bound = 0
    current_coord = path_coord
    while our_map[current_coord] != 2:
        if current_coord[ROW] + 1 >= num_row:
            break
        dist_to_south_bound += 1
        current_coord = (current_coord[COL], current_coord[ROW] + 1)

    dist_to_east_bound = 0
    current_coord = path_coord
    while our_map[current_coord] != 2:
        if current_coord[COL] + 1 >= num_col:
            break
        dist_to_east_bound += 1
        current_coord = (current_coord[COL] + 1, current_coord[ROW])

    dist_to_west_bound = 0
    current_coord = path_coord
    while our_map[current_coord] != 2:
        if current_coord[COL] - 1 < 0:
            break
        dist_to_west_bound += 1
        current_coord = (current_coord[COL] - 1, current_coord[ROW])

    return np.min([dist_to_north_bound, dist_to_west_bound, 
        dist_to_east_bound, dist_to_south_bound])


def get_path_cost(path, our_map):
    count_cross_lane_boundary = 0
    min_dist_from_lane_bound = []

    for i, p in enumerate(path):
        x = round(p[0])
        y = round(p[1])

        # if crossing a lane boundary
        if our_map[(x, y)] == 2:
            count_cross_lane_boundary += 1
        # if not crossing a lane boundary, get minimum distance from lane boundary
        else:
            d = dist_from_lane_bound((x, y), our_map)
            min_dist_from_lane_bound.append(d)

    return count_cross_lane_boundary - np.mean(min_dist_from_lane_bound) 
