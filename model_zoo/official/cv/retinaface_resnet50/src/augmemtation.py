# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Augmentation."""
import random
import copy
import cv2
import numpy as np

def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a

def bbox_iof(bbox_a, bbox_b, offset=0):

    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, 0:2], bbox_b[:, 0:2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    return area_i / np.maximum(area_a[:, None], 1)

def _is_iof_satisfied_constraint(box, crop_box):
    iof = bbox_iof(box, crop_box)
    satisfied = np.any((iof >= 1.0))
    return satisfied

def _choose_candidate(max_trial, image_w, image_h, boxes):
    # add default candidate
    candidates = [(0, 0, image_w, image_h)]

    for _ in range(max_trial):
        # box_data should have at least one box
        if _rand() > 0.2:
            scale = _rand(0.3, 1.0)
        else:
            scale = 1.0

        nh = int(scale * min(image_w, image_h))
        nw = nh

        dx = int(_rand(0, image_w - nw))
        dy = int(_rand(0, image_h - nh))

        if boxes.shape[0] > 0:
            crop_box = np.array((dx, dy, dx + nw, dy + nh))
            if not _is_iof_satisfied_constraint(boxes, crop_box[np.newaxis]):
                continue
            else:
                candidates.append((dx, dy, nw, nh))
        else:
            raise Exception("!!! annotation box is less than 1")

        if len(candidates) >= 3:
            break

    return candidates

def _correct_bbox_by_candidates(candidates, input_w, input_h, flip, boxes, labels, landms, allow_outside_center):
    """Calculate correct boxes."""
    while candidates:
        if len(candidates) > 1:
            # ignore default candidate which do not crop
            candidate = candidates.pop(np.random.randint(1, len(candidates)))
        else:
            candidate = candidates.pop(np.random.randint(0, len(candidates)))
        dx, dy, nw, nh = candidate

        boxes_t = copy.deepcopy(boxes)
        landms_t = copy.deepcopy(landms)
        labels_t = copy.deepcopy(labels)
        landms_t = landms_t.reshape([-1, 5, 2])

        if nw == nh:
            scale = float(input_w) / float(nw)
        else:
            scale = float(input_w) / float(max(nh, nw))
        boxes_t[:, [0, 2]] = (boxes_t[:, [0, 2]] - dx) * scale
        boxes_t[:, [1, 3]] = (boxes_t[:, [1, 3]] - dy) * scale
        landms_t[:, :, 0] = (landms_t[:, :, 0] - dx) * scale
        landms_t[:, :, 1] = (landms_t[:, :, 1] - dy) * scale

        if flip:
            boxes_t[:, [0, 2]] = input_w - boxes_t[:, [2, 0]]
            landms_t[:, :, 0] = input_w - landms_t[:, :, 0]
            # flip landms
            landms_t_1 = landms_t[:, 1, :].copy()
            landms_t[:, 1, :] = landms_t[:, 0, :]
            landms_t[:, 0, :] = landms_t_1
            landms_t_4 = landms_t[:, 4, :].copy()
            landms_t[:, 4, :] = landms_t[:, 3, :]
            landms_t[:, 3, :] = landms_t_4

        if allow_outside_center:
            pass
        else:
            mask1 = np.logical_and((boxes_t[:, 0] + boxes_t[:, 2])/2. >= 0., (boxes_t[:, 1] + boxes_t[:, 3])/2. >= 0.)
            boxes_t = boxes_t[mask1]
            landms_t = landms_t[mask1]
            labels_t = labels_t[mask1]

            mask2 = np.logical_and((boxes_t[:, 0] + boxes_t[:, 2]) / 2. <= input_w,
                                   (boxes_t[:, 1] + boxes_t[:, 3]) / 2. <= input_h)
            boxes_t = boxes_t[mask2]
            landms_t = landms_t[mask2]
            labels_t = labels_t[mask2]

        # recorrect x, y for case x,y < 0 reset to zero, after dx and dy, some box can smaller than zero
        boxes_t[:, 0:2][boxes_t[:, 0:2] < 0] = 0
        # recorrect w,h not higher than input size
        boxes_t[:, 2][boxes_t[:, 2] > input_w] = input_w
        boxes_t[:, 3][boxes_t[:, 3] > input_h] = input_h
        box_w = boxes_t[:, 2] - boxes_t[:, 0]
        box_h = boxes_t[:, 3] - boxes_t[:, 1]
        # discard invalid box: w or h smaller than 1 pixel
        mask3 = np.logical_and(box_w > 1, box_h > 1)
        boxes_t = boxes_t[mask3]
        landms_t = landms_t[mask3]
        labels_t = labels_t[mask3]

        # normal
        boxes_t[:, [0, 2]] /= input_w
        boxes_t[:, [1, 3]] /= input_h
        landms_t[:, :, 0] /= input_w
        landms_t[:, :, 1] /= input_h

        landms_t = landms_t.reshape([-1, 10])
        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((boxes_t, landms_t, labels_t))

        if boxes_t.shape[0] > 0:

            return targets_t, candidate

    raise Exception('all candidates can not satisfied re-correct bbox')

def get_interp_method(interp, sizes=()):
    """Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Nearest Neighbors. [Originally it should be Area-based,
        as we cannot find Area-based, so we use NN instead.
        Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method mentioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
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

def cv_image_reshape(interp):
    """Reshape pil image."""
    reshape_type = {
        0: cv2.INTER_LINEAR,
        1: cv2.INTER_CUBIC,
        2: cv2.INTER_AREA,
        3: cv2.INTER_NEAREST,
        4: cv2.INTER_LANCZOS4,
    }
    return reshape_type[interp]

def color_convert(image, a=1, b=0):
    c_image = image.astype(float) * a + b
    c_image[c_image < 0] = 0
    c_image[c_image > 255] = 255

    image[:] = c_image

def color_distortion(image):
    image = copy.deepcopy(image)

    if _rand() > 0.5:
        if _rand() > 0.5:
            color_convert(image, b=_rand(-32, 32))
        if _rand() > 0.5:
            color_convert(image, a=_rand(0.5, 1.5))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if _rand() > 0.5:
            color_convert(image[:, :, 1], a=_rand(0.5, 1.5))
        if _rand() > 0.5:
            h_img = image[:, :, 0].astype(int) + random.randint(-18, 18)
            h_img %= 180
            image[:, :, 0] = h_img
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    else:
        if _rand() > 0.5:
            color_convert(image, b=random.uniform(-32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if _rand() > 0.5:
            color_convert(image[:, :, 1], a=random.uniform(0.5, 1.5))
        if _rand() > 0.5:
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if _rand() > 0.5:
            color_convert(image, a=random.uniform(0.5, 1.5))

    return image

class preproc():
    def __init__(self, image_dim):
        self.image_input_size = image_dim

    def __call__(self, image, target):
        assert target.shape[0] > 0, "target without ground truth."
        _target = copy.deepcopy(target)
        boxes = _target[:, :4]
        landms = _target[:, 4:-1]
        labels = _target[:, -1]

        aug_image, aug_target = self._data_aug(image, boxes, labels, landms, self.image_input_size)

        return aug_image, aug_target

    def _data_aug(self, image, boxes, labels, landms, image_input_size, max_trial=250):


        image_h, image_w, _ = image.shape
        input_h, input_w = image_input_size, image_input_size

        flip = _rand() < .5

        candidates = _choose_candidate(max_trial=max_trial,
                                       image_w=image_w,
                                       image_h=image_h,
                                       boxes=boxes)
        targets, candidate = _correct_bbox_by_candidates(candidates=candidates,
                                                         input_w=input_w,
                                                         input_h=input_h,
                                                         flip=flip,
                                                         boxes=boxes,
                                                         labels=labels,
                                                         landms=landms,
                                                         allow_outside_center=False)
        # crop image
        dx, dy, nw, nh = candidate
        image = image[dy:(dy + nh), dx:(dx + nw)]

        if nw != nh:
            assert nw == image_w and nh == image_h
            # pad ori image to square
            l = max(nw, nh)
            t_image = np.empty((l, l, 3), dtype=image.dtype)
            t_image[:, :] = (104, 117, 123)
            t_image[:nh, :nw] = image
            image = t_image

        interp = get_interp_method(interp=10)
        image = cv2.resize(image, (input_w, input_h), interpolation=cv_image_reshape(interp))

        if flip:
            image = image[:, ::-1]

        image = image.astype(np.float32)
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)

        return image, targets
