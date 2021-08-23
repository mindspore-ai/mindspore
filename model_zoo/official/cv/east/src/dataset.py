# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import math
import os

from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image

import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as CV
from src.distributed_sampler import DistributedSampler


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
      vertices: vertices of text region <numpy.ndarray, (8,)>
      index1  : offset of point1
      index2  : offset of point2
      r       : [r1, r2, r3, r4] in paper
      coef    : shrink ratio in paper
    Output:
      vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(
        vertices[x1_index],
        vertices[y1_index],
        vertices[x2_index],
        vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
      vertices: vertices of text region <numpy.ndarray, (8,)>
      coef    : shrink ratio in paper
    Output:
      v       : vertices of shrunk text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
            cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
      vertices: vertices of text region <numpy.ndarray, (8,)>
      theta   : angle in radian measure
      anchor  : fixed position during rotation
    Output:
      rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
      vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
      the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
      vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
      err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + \
          cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + \
          cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
      vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
      the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(
        list(
            range(
                len(area_list))),
        key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
      start_loc: left-top position
      length   : length of crop image
      vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
      True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w +
                  length, start_h, start_w +
                  length, start_h +
                  length, start_w, start_h +
                  length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
      img         : PIL Image
      vertices    : vertices of text regions <numpy.ndarray, (n,8)>
      labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
      length      : length of cropped image region
    Output:
      region      : cropped image region
      new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert (ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length,
                             new_vertices[labels == 1, :])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
      rotate_mat: rotatation matrix
      anchor_x  : fixed x position
      anchor_y  : fixed y position
      length    : length of image
    Output:
      rotated_x : rotated x positions <numpy.ndarray, (length,length)>
      rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.matmul(rotate_mat.astype(np.float16),
                              (coord_mat - np.array([[anchor_x],
                                                     [anchor_y]])).astype(np.float16)) + np.array([[anchor_x],
                                                                                                   [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
      img         : PIL Image
      vertices    : vertices of text regions <numpy.ndarray, (n,8)>
      ratio       : height changes in [0.8, 1.2]
    Output:
      img         : adjusted PIL Image
      new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
      img         : PIL Image
      vertices    : vertices of text regions <numpy.ndarray, (n,8)>
      angle_range : rotate range
    Output:
      img         : rotated PIL Image
      new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(
            vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return img, new_vertices


def get_score_geo(img, vertices, labels, scale, length):
    '''generate score gt and geometry gt
    Input:
      img     : PIL Image
      vertices: vertices of text regions <numpy.ndarray, (n,8)>
      labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
      scale   : feature map / image
      length  : image length
    Output:
      score gt, geo gt, ignored
    '''
    score_map = np.zeros(
        (int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros(
        (int(img.height * scale), int(img.width * scale), 5), np.float32)
    ignored_map = np.zeros(
        (int(img.height * scale), int(img.width * scale), 1), np.float32)

    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue

        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask

    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)
    return score_map, geo_map, ignored_map


def extract_vertices(lines):
    '''extract vertices info from txt lines
    Input:
      lines   : list of string info
    Output:
      vertices: vertices of text regions <numpy.ndarray, (n,8)>
      labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
    '''
    labels = []
    vertices = []
    for line in lines:
        vertices.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 0 if '###' in line else 1
        labels.append(label)
    return np.array(vertices), np.array(labels)


class ICDAREASTDataset:
    def __init__(self, img_path, gt_path, scale=0.25, length=512):
        super(ICDAREASTDataset, self).__init__()
        self.img_files = [os.path.join(
            img_path,
            img_file) for img_file in sorted(os.listdir(img_path))]
        self.gt_files = [
            os.path.join(
                gt_path,
                gt_file) for gt_file in sorted(
                    os.listdir(gt_path))]
        self.scale = scale
        self.length = length

    def __getitem__(self, index):
        with open(self.gt_files[index], 'r') as f:
            lines = f.readlines()
        vertices, labels = extract_vertices(lines)

        img = Image.open(self.img_files[index])
        img, vertices = adjust_height(img, vertices)
        img, vertices = rotate_img(img, vertices)
        img, vertices = crop_img(img, vertices, labels, self.length)
        score_map, geo_map, ignored_map = get_score_geo(
            img, vertices, labels, self.scale, self.length)
        score_map = score_map.transpose(2, 0, 1)
        ignored_map = ignored_map.transpose(2, 0, 1)
        geo_map = geo_map.transpose(2, 0, 1)
        if np.sum(score_map) < 1:
            score_map[0, 0, 0] = 1
        return img, score_map, geo_map, ignored_map

    def __len__(self):
        return len(self.img_files)


def create_east_dataset(
        img_root,
        txt_root,
        batch_size,
        device_num,
        rank,
        is_training=True):
    east_data = ICDAREASTDataset(img_path=img_root, gt_path=txt_root)
    distributed_sampler = DistributedSampler(
        len(east_data), device_num, rank, shuffle=True)

    trans_list = [CV.RandomColorAdjust(0.5, 0.5, 0.5, 0.25),
                  CV.Rescale(1 / 255.0, 0),
                  CV.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  CV.HWC2CHW()]
    if is_training:
        dataset_column_names = [
            "image",
            "score_map",
            "geo_map",
            "training_mask"]
        ds = de.GeneratorDataset(
            east_data,
            column_names=dataset_column_names,
            num_parallel_workers=32,
            sampler=distributed_sampler)
        ds = ds.map(
            operations=trans_list,
            input_columns=["image"],
            num_parallel_workers=8,
            python_multiprocessing=True)
        ds = ds.batch(batch_size, num_parallel_workers=8, drop_remainder=True)

    return ds, len(east_data)
