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
"""Preprocess dataset."""
import random
import threading
import copy

import numpy as np
from PIL import Image
import cv2
import mindspore.dataset.vision.py_transforms as PV


def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def statistic_normalize_img(img, statistic_norm):
    """Statistic normalize images."""
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    return img


def get_interp_method(interp, sizes=()):
    """
    Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic or Bilinear.

    Args:
        interp (int): Interpolation method for all resizing operations.

            - 0: Nearest Neighbors Interpolation.
            - 1: Bilinear interpolation.
            - 2: Bicubic interpolation over 4x4 pixel neighborhood.
            - 3: Nearest Neighbors. Originally it should be Area-based, as we cannot find Area-based,
              so we use NN instead. Area-based (resampling using pixel area relation).
              It may be a preferred method for image decimation, as it gives moire-free results.
              But when the image is zoomed, it is similar to the Nearest Neighbors method. (used by default).
            - 4: Lanczos interpolation over 8x8 pixel neighborhood.
            - 9: Cubic for enlarge, area for shrink, bilinear for others.
            - 10: Random select from interpolation method mentioned above.

        sizes (tuple): Format should like (old_height, old_width, new_height, new_width),
            if None provided, auto(9) will return Area(2) anyway. Default: ()

    Returns:
        int, interp method from 0 to 4.
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            if nh < oh and nw < ow:
                return 0
            return 1
        return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def pil_image_reshape(interp):
    """Reshape pil image."""
    reshape_type = {
        0: Image.NEAREST,
        1: Image.BILINEAR,
        2: Image.BICUBIC,
        3: Image.NEAREST,
        4: Image.LANCZOS,
    }
    return reshape_type[interp]


def _preprocess_true_boxes(true_boxes, anchors, in_shape, num_classes, max_boxes, label_smooth,
                           label_smooth_factor=0.1, iou_threshold=0.213):
    """
    Introduction
    ------------
        preprocessing ground truth box
    Parameters
    ----------
        true_boxes: ground truth box shape as [boxes, 5], x_min, y_min, x_max, y_max, class_id
    """
    anchors = np.array(anchors)
    num_layers = anchors.shape[0] // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(in_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
    # trans to box center point
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # input_shape is [h, w]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
    # true_boxes = [xywh]
    grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
    # grid_shape [h, w]
    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
                        5 + num_classes), dtype='float32') for l in range(num_layers)]
    # y_true [gridy, gridx]
    anchors = np.expand_dims(anchors, 0)
    anchors_max = anchors / 2.
    anchors_min = -anchors_max
    valid_mask = boxes_wh[..., 0] > 0
    wh = boxes_wh[valid_mask]
    if wh.size != 0:
        wh = np.expand_dims(wh, -2)
        # wh shape[box_num, 1, 2]
        boxes_max = wh / 2.
        boxes_min = -boxes_max
        intersect_min = np.maximum(boxes_min, anchors_min)
        intersect_max = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # topk iou
        topk = 4
        topk_flag = iou.argsort()
        topk_flag = topk_flag >= topk_flag.shape[1] - topk
        flag = topk_flag.nonzero()
        for index in range(len(flag[0])):
            t = flag[0][index]
            n = flag[1][index]
            if iou[t][n] < iou_threshold:
                continue
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')  # grid_y
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')  # grid_x

                    k = anchor_mask[l].index(n)
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1.

                    # lable-smooth
                    if label_smooth:
                        sigma = label_smooth_factor / (num_classes - 1)
                        y_true[l][j, i, k, 5:] = sigma
                        y_true[l][j, i, k, 5 + c] = 1 - label_smooth_factor
                    else:
                        y_true[l][j, i, k, 5 + c] = 1.
        # best anchor for gt
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')  # grid_y
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')  # grid_x

                    k = anchor_mask[l].index(n)
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1.

                    # lable-smooth
                    if label_smooth:
                        sigma = label_smooth_factor / (num_classes - 1)
                        y_true[l][j, i, k, 5:] = sigma
                        y_true[l][j, i, k, 5 + c] = 1 - label_smooth_factor
                    else:
                        y_true[l][j, i, k, 5 + c] = 1.

    # pad_gt_boxes for avoiding dynamic shape
    pad_gt_box0 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)
    pad_gt_box1 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)
    pad_gt_box2 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)

    mask0 = np.reshape(y_true[0][..., 4:5], [-1])
    gt_box0 = np.reshape(y_true[0][..., 0:4], [-1, 4])
    # gt_box [boxes, [x,y,w,h]]
    gt_box0 = gt_box0[mask0 == 1]
    # gt_box0: get all boxes which have object
    if gt_box0.shape[0] < max_boxes:
        pad_gt_box0[:gt_box0.shape[0]] = gt_box0
    else:
        pad_gt_box0 = gt_box0[:max_boxes]
    # gt_box0.shape[0]: total number of boxes in gt_box0
    # top N of pad_gt_box0 is real box, and after are pad by zero

    mask1 = np.reshape(y_true[1][..., 4:5], [-1])
    gt_box1 = np.reshape(y_true[1][..., 0:4], [-1, 4])
    gt_box1 = gt_box1[mask1 == 1]
    if gt_box1.shape[0] < max_boxes:
        pad_gt_box1[:gt_box1.shape[0]] = gt_box1
    else:
        pad_gt_box1 = gt_box1[:max_boxes]

    mask2 = np.reshape(y_true[2][..., 4:5], [-1])
    gt_box2 = np.reshape(y_true[2][..., 0:4], [-1, 4])

    gt_box2 = gt_box2[mask2 == 1]
    if gt_box2.shape[0] < max_boxes:
        pad_gt_box2[:gt_box2.shape[0]] = gt_box2
    else:
        pad_gt_box2 = gt_box2[:max_boxes]
    return y_true[0], y_true[1], y_true[2], pad_gt_box0, pad_gt_box1, pad_gt_box2


class PreprocessTrueBox:
    def __init__(self, config):
        self.anchor_scales = config.anchor_scales
        self.num_classes = config.num_classes
        self.max_box = config.max_box
        self.label_smooth = config.label_smooth
        self.label_smooth_factor = config.label_smooth_factor

    def __call__(self, anno, input_shape):
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(true_boxes=anno, anchors=self.anchor_scales, in_shape=input_shape,
                                   num_classes=self.num_classes, max_boxes=self.max_box,
                                   label_smooth=self.label_smooth, label_smooth_factor=self.label_smooth_factor)
        return anno, np.array(bbox_true_1), np.array(bbox_true_2), np.array(bbox_true_3), \
               np.array(gt_box1), np.array(gt_box2), np.array(gt_box3)


def _reshape_data(image, image_size):
    """Reshape image."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    ori_w, ori_h = image.size
    ori_image_shape = np.array([ori_w, ori_h], np.int32)
    # original image shape fir:H sec:W
    h, w = image_size
    interp = get_interp_method(interp=9, sizes=(ori_h, ori_w, h, w))
    image = image.resize((w, h), pil_image_reshape(interp))
    image_data = statistic_normalize_img(image, statistic_norm=True)
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
    image_data = image_data.astype(np.float32)
    return image_data, ori_image_shape


def color_distortion(img, hue, sat, val, device_num):
    """Color distortion."""
    hue = _rand(-hue, hue)
    sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
    val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
    if device_num != 1:
        cv2.setNumThreads(1)
    x = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    x = x / 255.
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    x = x * 255.
    x = x.astype(np.uint8)
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB_FULL)
    return image_data


def filp_pil_image(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def convert_gray_to_color(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        img = np.concatenate([img, img, img], axis=-1)
    return img


def _is_iou_satisfied_constraint(min_iou, max_iou, box, crop_box):
    iou = bbox_iou(box, crop_box)
    return min_iou <= iou.min() and max_iou >= iou.max()


def _choose_candidate_by_constraints(max_trial, input_w, input_h, image_w, image_h, jitter, box, use_constraints):
    """Choose candidate by constraints."""
    if use_constraints:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )
    else:
        constraints = ((None, None),)
    # add default candidate
    candidates = [(0, 0, input_w, input_h)]
    for constraint in constraints:
        min_iou, max_iou = constraint
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            # box_data should have at least one box
            new_ar = float(input_w) / float(input_h) * _rand(1 - jitter, 1 + jitter) / _rand(1 - jitter, 1 + jitter)
            scale = _rand(0.5, 2)

            if new_ar < 1:
                nh = int(scale * input_h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * input_w)
                nh = int(nw / new_ar)

            dx = int(_rand(0, input_w - nw))
            dy = int(_rand(0, input_h - nh))

            if box.size > 0:
                t_box = copy.deepcopy(box)
                t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(image_w) + dx
                t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(image_h) + dy

                crop_box = np.array((0, 0, input_w, input_h))
                if not _is_iou_satisfied_constraint(min_iou, max_iou, t_box, crop_box[np.newaxis]):
                    continue
                else:
                    candidates.append((dx, dy, nw, nh))
            else:
                raise Exception("!!! annotation box is less than 1")
    return candidates


def _correct_bbox_by_candidates(candidates, input_w, input_h, image_w,
                                image_h, flip, box, box_data, allow_outside_center, max_boxes):
    """Calculate correct boxes."""
    while candidates:
        if len(candidates) > 1:
            # ignore default candidate which do not crop
            candidate = candidates.pop(np.random.randint(1, len(candidates)))
        else:
            candidate = candidates.pop(np.random.randint(0, len(candidates)))
        dx, dy, nw, nh = candidate
        t_box = copy.deepcopy(box)
        t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(image_w) + dx
        t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(image_h) + dy
        if flip:
            t_box[:, [0, 2]] = input_w - t_box[:, [2, 0]]

        if allow_outside_center:
            pass
        else:
            t_box = t_box[
                np.logical_and((t_box[:, 0] + t_box[:, 2]) / 2. >= 0., (t_box[:, 1] + t_box[:, 3]) / 2. >= 0.)]
            t_box = t_box[np.logical_and((t_box[:, 0] + t_box[:, 2]) / 2. <= input_w,
                                         (t_box[:, 1] + t_box[:, 3]) / 2. <= input_h)]

        # recorrect x, y for case x,y < 0 reset to zero, after dx and dy, some box can smaller than zero
        t_box[:, 0:2][t_box[:, 0:2] < 0] = 0
        # recorrect w,h not higher than input size
        t_box[:, 2][t_box[:, 2] > input_w] = input_w
        t_box[:, 3][t_box[:, 3] > input_h] = input_h
        box_w = t_box[:, 2] - t_box[:, 0]
        box_h = t_box[:, 3] - t_box[:, 1]
        # discard invalid box: w or h smaller than 1 pixel
        t_box = t_box[np.logical_and(box_w > 1, box_h > 1)]

        if t_box.shape[0] > 0:
            # break if number of find t_box
            box_data[: len(t_box)] = t_box
            return box_data, candidate
    return np.zeros(shape=[max_boxes, 5], dtype=np.float64), (0, 0, nw, nh)


def _data_aug(image, box, jitter, hue, sat, val, image_input_size, max_boxes,
              anchors, num_classes, max_trial=10, device_num=1):
    """Crop an image randomly with bounding box constraints.

        This data augmentation is used in training of
        Single Shot Multibox Detector [#]_. More details can be found in
        data augmentation section of the original paper.
        .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
           Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
           SSD: Single Shot MultiBox Detector. ECCV 2016."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_w, image_h = image.size
    input_h, input_w = image_input_size

    np.random.shuffle(box)
    if len(box) > max_boxes:
        box = box[:max_boxes]
    flip = _rand() < .5
    box_data = np.zeros((max_boxes, 5))

    candidates = _choose_candidate_by_constraints(use_constraints=False, max_trial=max_trial, input_w=input_w,
                                                  input_h=input_h, image_w=image_w, image_h=image_h,
                                                  jitter=jitter, box=box)
    box_data, candidate = _correct_bbox_by_candidates(candidates=candidates, input_w=input_w, input_h=input_h,
                                                      image_w=image_w, image_h=image_h, flip=flip, box=box,
                                                      box_data=box_data, allow_outside_center=True, max_boxes=max_boxes)
    dx, dy, nw, nh = candidate
    interp = get_interp_method(interp=10)
    image = image.resize((nw, nh), pil_image_reshape(interp))
    # place image, gray color as back graoud
    new_image = Image.new('RGB', (input_w, input_h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    if flip:
        image = filp_pil_image(image)

    image = np.array(image)
    image = convert_gray_to_color(image)
    image_data = color_distortion(image, hue, sat, val, device_num)
    return image_data, box_data


def preprocess_fn(image, box, config, input_size, device_num):
    """Preprocess data function."""
    config_anchors = config.anchor_scales
    anchors = np.array([list(x) for x in config_anchors])
    max_boxes = config.max_box
    num_classes = config.num_classes
    jitter = config.jitter
    hue = config.hue
    sat = config.saturation
    val = config.value
    image, anno = _data_aug(image, box, jitter=jitter, hue=hue, sat=sat, val=val,
                            image_input_size=input_size, max_boxes=max_boxes,
                            num_classes=num_classes, anchors=anchors, device_num=device_num)
    return image, anno


def reshape_fn(image, img_id, config):
    input_size = config.test_img_shape
    image, ori_image_shape = _reshape_data(image, image_size=input_size)
    return image, ori_image_shape, img_id


class MultiScaleTrans:
    """Multi scale transform."""

    def __init__(self, config, device_num):
        self.config = config
        self.seed = 0
        self.size_list = []
        self.resize_rate = config.resize_rate
        self.dataset_size = config.dataset_size
        self.size_dict = {}
        self.seed_num = int(1e6)
        self.seed_list = self.generate_seed_list(seed_num=self.seed_num)
        self.resize_count_num = int(np.ceil(self.dataset_size / self.resize_rate))
        self.device_num = device_num
        self.anchor_scales = config.anchor_scales
        self.num_classes = config.num_classes
        self.max_box = config.max_box
        self.label_smooth = config.label_smooth
        self.label_smooth_factor = config.label_smooth_factor

    def generate_seed_list(self, init_seed=1234, seed_num=int(1e6), seed_range=(1, 1000)):
        seed_list = []
        random.seed(init_seed)
        for _ in range(seed_num):
            seed = random.randint(seed_range[0], seed_range[1])
            seed_list.append(seed)
        return seed_list

    def __call__(self, img, anno, input_size, mosaic_flag):
        if mosaic_flag[0] == 0:
            img = PV.Decode()(img)
        img, anno = preprocess_fn(img, anno, self.config, input_size, self.device_num)
        return img, anno, np.array(img.shape[0:2])


def thread_batch_preprocess_true_box(annos, config, input_shape, result_index, batch_bbox_true_1, batch_bbox_true_2,
                                     batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3):
    """Preprocess true box for multi-thread."""
    i = 0
    for anno in annos:
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(true_boxes=anno, anchors=config.anchor_scales, in_shape=input_shape,
                                   num_classes=config.num_classes, max_boxes=config.max_box,
                                   label_smooth=config.label_smooth, label_smooth_factor=config.label_smooth_factor)
        batch_bbox_true_1[result_index + i] = bbox_true_1
        batch_bbox_true_2[result_index + i] = bbox_true_2
        batch_bbox_true_3[result_index + i] = bbox_true_3
        batch_gt_box1[result_index + i] = gt_box1
        batch_gt_box2[result_index + i] = gt_box2
        batch_gt_box3[result_index + i] = gt_box3
        i = i + 1


def batch_preprocess_true_box(annos, config, input_shape):
    """Preprocess true box with multi-thread."""
    batch_bbox_true_1 = []
    batch_bbox_true_2 = []
    batch_bbox_true_3 = []
    batch_gt_box1 = []
    batch_gt_box2 = []
    batch_gt_box3 = []
    threads = []

    step = 4
    for index in range(0, len(annos), step):
        for _ in range(step):
            batch_bbox_true_1.append(None)
            batch_bbox_true_2.append(None)
            batch_bbox_true_3.append(None)
            batch_gt_box1.append(None)
            batch_gt_box2.append(None)
            batch_gt_box3.append(None)
        step_anno = annos[index: index + step]
        t = threading.Thread(target=thread_batch_preprocess_true_box,
                             args=(step_anno, config, input_shape, index, batch_bbox_true_1, batch_bbox_true_2,
                                   batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return np.array(batch_bbox_true_1), np.array(batch_bbox_true_2), np.array(batch_bbox_true_3), \
           np.array(batch_gt_box1), np.array(batch_gt_box2), np.array(batch_gt_box3)


def batch_preprocess_true_box_single(annos, config, input_shape):
    """Preprocess true boxes."""
    batch_bbox_true_1 = []
    batch_bbox_true_2 = []
    batch_bbox_true_3 = []
    batch_gt_box1 = []
    batch_gt_box2 = []
    batch_gt_box3 = []
    for anno in annos:
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(true_boxes=anno, anchors=config.anchor_scales, in_shape=input_shape,
                                   num_classes=config.num_classes, max_boxes=config.max_box,
                                   label_smooth=config.label_smooth, label_smooth_factor=config.label_smooth_factor)
        batch_bbox_true_1.append(bbox_true_1)
        batch_bbox_true_2.append(bbox_true_2)
        batch_bbox_true_3.append(bbox_true_3)
        batch_gt_box1.append(gt_box1)
        batch_gt_box2.append(gt_box2)
        batch_gt_box3.append(gt_box3)

    return np.array(batch_bbox_true_1), np.array(batch_bbox_true_2), np.array(batch_bbox_true_3), \
           np.array(batch_gt_box1), np.array(batch_gt_box2), np.array(batch_gt_box3)
