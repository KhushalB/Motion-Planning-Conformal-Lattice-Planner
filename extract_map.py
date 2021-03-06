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
