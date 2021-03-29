import math
import pandas as pd
import json

from cruw.mapping.object_types import get_class_id
from cruw.mapping.ops import find_nearest


def read_dets_kitti_txt(txt_path, dataset):
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid
    classes = dataset.object_cfg.classes
    t_cl2rh = dataset.sensor_cfg.calib_cfg['t_cl2rh']

    label = open(txt_path)
    obj_info = []

    for line in label:
        # for each object
        line = line.rstrip().split()
        class_str = line[0]
        class_id = get_class_id(class_str, classes)
        trunc = line[1]
        visi = line[2]
        bbox_x = float(line[4])
        bbox_y = float(line[5])
        bbox_w = float(line[6]) - bbox_x
        bbox_h = float(line[7]) - bbox_y

        x = float(line[11]) - t_cl2rh[0]
        y = float(line[12]) - t_cl2rh[1]
        z = float(line[13]) - t_cl2rh[2]
        score = float(line[15])
        distance = math.sqrt(x ** 2 + z ** 2)
        angle = math.atan(x / z)  # radians
        if distance > dataset.sensor_cfg.radar_cfg['rr_max'] or \
                distance < dataset.sensor_cfg.radar_cfg['rr_min']:
            # ignore the objects out of the range
            continue
        if angle > math.radians(dataset.sensor_cfg.radar_cfg['ra_max']) or \
                angle < math.radians(dataset.sensor_cfg.radar_cfg['ra_min']):
            # ignore the objects out of the range
            continue
        rng_idx, _ = find_nearest(range_grid, distance)
        agl_idx, _ = find_nearest(angle_grid, angle)
        obj_dict = dict(
            type=class_str,
            class_id=class_id,
            bbox=[bbox_x, bbox_y, bbox_w, bbox_h],
            trans=[x, y, z],
            ra_id=[rng_idx, agl_idx],
            ra=[distance, angle],
            score=score,
            source='camera',
            visi=visi,
            trunc=trunc
        )
        obj_info.append(obj_dict)

    return obj_info


def read_ra_labels_csv(csv_path, dataset):
    data = pd.read_csv(csv_path)
    n_row, n_col = data.shape

    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid
    range_grid_label = dataset.range_grid_label
    angle_grid_label = dataset.angle_grid_label
    classes = dataset.object_cfg.classes

    obj_info_list = []
    cur_idx = -1
    for r in range(n_row):
        filename = data['filename'][r]
        frame_idx = int(filename.split('.')[0].split('_')[-1])
        if cur_idx == -1:
            obj_info = []
            cur_idx = frame_idx
        if frame_idx > cur_idx:
            obj_info_list.append(obj_info)
            obj_info = []
            cur_idx = frame_idx

        region_count = data['region_count'][r]
        region_id = data['region_id'][r]

        if region_count != 0:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])

            cx = region_shape_attri['cx']
            cy = region_shape_attri['cy']
            distance = range_grid_label[cy]
            angle = angle_grid_label[cx]  # radians
            if distance > dataset.sensor_cfg.radar_cfg['rr_max'] or \
                    distance < dataset.sensor_cfg.radar_cfg['rr_min']:
                continue
            if angle > math.radians(dataset.sensor_cfg.radar_cfg['ra_max']) or \
                    angle < math.radians(dataset.sensor_cfg.radar_cfg['ra_min']):
                continue
            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)
            try:
                class_str = region_attri['class']
            except:
                print("missing class at row %d" % r)
                continue
            class_id = get_class_id(class_str, classes)
            obj_dict = dict(
                type=class_str,
                class_id=class_id,
                ra_id=[int(rng_idx), int(agl_idx)],
                ra=[distance, angle],
                source='human',
            )
            obj_info.append(obj_dict)

    obj_info_list.append(obj_info)

    return obj_info_list


def read_dets_crf_txt(txt_path, dataset):
    classes = dataset.object_cfg.classes

    label = open(txt_path)
    obj_info = []

    for line in label:
        # for each object
        line = line.rstrip().split()
        class_str = line[1]
        class_id = get_class_id(class_str, classes)

        rng_idx = float(line[2])
        agl_idx = float(line[4])
        distance = float(line[3])
        angle = float(line[5])  # radians
        score = float(line[6])
        if distance > dataset.sensor_cfg.radar_cfg['rr_max'] or \
                distance < dataset.sensor_cfg.radar_cfg['rr_min']:
            # ignore the objects out of the range
            continue
        if angle > math.radians(dataset.sensor_cfg.radar_cfg['ra_max']) or \
                angle < math.radians(dataset.sensor_cfg.radar_cfg['ra_min']):
            # ignore the objects out of the range
            continue

        obj_dict = dict(
            type=class_str,
            class_id=class_id,
            ra_id=[rng_idx, agl_idx],
            ra=[distance, angle],
            score=score,
            source='crf',
        )
        obj_info.append(obj_dict)

    return obj_info


def read_image_labels_csv(csv_path):
    data = pd.read_csv(csv_path)
    n_row, n_col = data.shape

    obj_info_list = []
    cur_idx = -1
    for r in range(n_row):
        filename = data['filename'][r]
        frame_idx = int(filename.split('.')[0].split('_')[-1])
        if cur_idx == -1:
            obj_info = []
            cur_idx = frame_idx
        if frame_idx > cur_idx:
            obj_info_list.append(obj_info)
            obj_info = []
            cur_idx = frame_idx

        region_count = data['region_count'][r]
        region_id = data['region_id'][r]

        if region_count != 0:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])

            bbox_x = region_shape_attri['x']
            bbox_y = region_shape_attri['y']
            bbox_w = region_shape_attri['width']
            bbox_h = region_shape_attri['height']
            visibilities = region_attri['occlusion']
            truncations = region_attri['truncation']
            # reachability = region_attri['reachability']
            try:
                class_str = region_attri['class']
            except:
                print("missing class at row %d" % r)
                continue
            obj_dict = dict(
                type=class_str,
                bbox=[bbox_x, bbox_y, bbox_w, bbox_h],
                source='human',
                visi=visibilities,
                trunc=truncations,

            )
            obj_info.append(obj_dict)

    obj_info_list.append(obj_info)

    return obj_info_list
