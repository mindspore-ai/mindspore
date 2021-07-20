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
"""util."""
import sys
import numpy as np
import cv2
from PIL import Image

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def write_depth(path, depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        :param path:
        :param depth:
        :param bits:
    """
    write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return out


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): path file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
                len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def depth_read_kitti(filename):
    """

    :type filename: object
    """
    depth_png = np.array(Image.open(filename), dtype=int)
    assert np.max(depth_png) > 255

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth


def depth_read_sintel(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file ' \
                               '(should be: {0}, is: {1}). Big-endian machine? ' \
        .format(TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert width > 0 and height > 0 and 1 < size < 100000000, \
        ' depth_read:: Wrong input size (width = {0}, height = {1}).' \
        .format(width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


class BadPixelMetric:
    """ BadPixelMetric. """

    def __init__(self, threshold=1.25, depth_cap=10, model='NYU'):
        self.__threshold = threshold
        self.__depth_cap = depth_cap
        self.__model = model

    def compute_scale_and_shift(self, prediction, target, mask):
        """ compute_scale_and_shift. """
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = np.sum(mask * prediction * prediction, (1, 2))
        a_01 = np.sum(mask * prediction, (1, 2))
        a_11 = np.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = np.sum(mask * prediction * target, (1, 2))
        b_1 = np.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):
        # transform predicted disparity to aligned depth
        target_disparity = np.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]

        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        prediction_aligned = scale.reshape((-1, 1, 1)) * prediction + shift.reshape((-1, 1, 1))

        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediciton_depth = 1.0 / prediction_aligned

        # bad pixel
        err = np.zeros_like(prediciton_depth, dtype=np.float)
        if self.__model == 'NYU' or self.__model == 'TUM' or self.__model == 'KITTI':
            err[mask == 1] = np.maximum(
                prediciton_depth[mask == 1] / target[mask == 1],
                target[mask == 1] / prediciton_depth[mask == 1],
            )

            err[mask == 1] = (err[mask == 1] > self.__threshold)

            p = np.sum(err, (1, 2)) / np.sum(mask, (1, 2))
        if self.__model == 'sintel' or self.__model == 'ETH3D':
            err[mask == 1] = np.abs((prediciton_depth[mask == 1] - target[mask == 1]) / target[mask == 1])
            err_sum = np.sum(err, (1, 2))
            mask_sum = np.sum(mask, (1, 2))
            print('err_sum is ', err_sum)
            print('mask_sum is ', mask_sum)
            if mask_sum == 0:
                p = np.zeros(1)
            else:
                p = err_sum / mask_sum
            return np.mean(p)

        return 100 * np.mean(p)
