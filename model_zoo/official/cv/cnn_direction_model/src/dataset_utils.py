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

from __future__ import absolute_import, division, print_function, unicode_literals

from math import ceil, sin, pi
from random import choice, random
from random import randint, uniform

import cv2
import numpy as np
from numpy.random import randn
from PIL import ImageEnhance, Image
from scipy.ndimage import filters, interpolation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import PiecewiseAffineTransform, warp

nprandint = np.random.randint


def lucky(p=0.3, rand_func=random):
    """ return True with probability p """
    return rand_func() < p


def rgeometry(im, eps=0.04, delta=0.8, cval=None, severity=1):
    """
    affine transform
    """
    if severity == 0:
        return im

    if cval is None:
        cval = [0] * im.shape[2]
    elif isinstance(cval, (float, int)):
        cval = [cval] * im.shape[2]

    severity = abs(severity)
    eps = severity * eps
    delta = severity * delta
    m = np.array([[1 + eps * randn(), 0.0], [eps * randn(), 1.0 + eps * randn()]])
    c = np.array(im.shape[:2]) * 0.5
    d = c - np.dot(m, c) + np.array([randn() * delta, randn() * delta])

    im = cv2.split(im)
    im = [interpolation.affine_transform(i, m, offset=d, order=1, mode='constant', cval=cval[e])
          for e, i in enumerate(im)]
    im = cv2.merge(im)

    return np.array(im)


def rdistort(im, distort=4.0, dsigma=10.0, cval=None, severity=1):
    """distort"""
    if severity == 0:
        return im

    if cval is None:
        cval = [0] * im.shape[2]
    elif isinstance(cval, (float, int)):
        cval = [cval] * im.shape[2]

    severity = abs(severity)
    distort = severity * distort
    dsigma = dsigma * (1 - severity)

    h, w = im.shape[:2]
    hs, ws = randn(h, w), randn(h, w)
    hs = filters.gaussian_filter(hs, dsigma)
    ws = filters.gaussian_filter(ws, dsigma)
    hs *= distort / np.abs(hs).max()
    ws *= distort / np.abs(ws).max()
    # When "ij" is passed in, the first array determines the column, the second array determines the row, by default,
    # the first array determines the row, and the second array determines the column
    ch, cw = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coordinates = np.array([ch + hs, cw + ws])

    im = cv2.split(im)
    im = [map_coordinates(img, coordinates, order=1, cval=cval[i]) for i, img in enumerate(im)]
    im = cv2.merge(im)
    return np.array(im)


def reverse_color(im):
    """ Pixel inversion """
    return 255 - im


def resize(im, fx=None, fy=None, delta=0.3, severity=1):
    """ scaling in the two directions of width fx and height fy,
    If the zoom factor is not specified, the maximum change amount of 0.3 is randomly selected from 1 to 1"""

    if fx is None:
        fx = 1 + delta * severity * uniform(-1, 1)
    if fy is None:
        fy = 1 + delta * severity * uniform(-1, 1)
    return np.array(cv2.resize(im, None, fx=fx, fy=fy))


def warp_perspective(im, theta=20, delta=10, cval=0, severity=1):
    """ perspective mapping """
    if severity == 0:
        return im

    if cval is None:
        cval = [0] * im.shape[2]
    elif isinstance(cval, (float, int)):
        cval = [cval] * im.shape[2]

    delta = delta * severity
    rows, cols = im.shape[:2]
    pts_im = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])

    # Distort randomly and constrain the scope of change
    pts_warp = pts_im + np.random.uniform(-1, 1, pts_im.shape) * theta * severity
    pts_warp = np.maximum(pts_warp, delta)  # Constrain the change to the part >=3
    pts_warp[[1, 2], 0] = np.minimum(pts_warp[[1, 2], 0], pts_im[[1, 2], 0] - delta)
    pts_warp[[2, 3], 1] = np.minimum(pts_warp[[2, 3], 1], pts_im[[2, 3], 1] - delta)
    pts_warp = np.float32(pts_warp)

    M = cv2.getPerspectiveTransform(pts_im, pts_warp)
    res = np.array(cv2.warpPerspective(im, M, (cols, rows), borderValue=cval))

    return res


def noise_salt_pepper(image, percentage=0.001, severity=1):
    """ Salt and pepper noise, percentage represents the percentage of salt and pepper noise"""
    percentage *= severity
    amount = int(percentage * image.shape[0] * image.shape[1])
    if amount == 0:
        return image
    _, _, deep = image.shape
    # Salt mode
    coords = [np.random.randint(0, i - 1, amount) for i in image.shape[:2]]
    salt = nprandint(200, 255, amount)
    salt = salt.repeat(deep, axis=0)
    image[coords[0], coords[1], :] = salt.reshape(amount, deep)

    # pepper mode
    coords = [np.random.randint(0, i - 1, amount) for i in image.shape[:2]]
    pepper = nprandint(0, 50, amount)
    pepper = pepper.repeat(deep, axis=0)
    image[coords[0], coords[1], :] = pepper.reshape(amount, deep)
    return image


def noise_gaussian(im, sigma=20, severity=1):
    """ add Gaussian noise"""
    sigma = sigma * abs(severity)
    return cvt_uint8(np.float32(im) + sigma * np.random.randn(*im.shape))


def noise_gamma(im, extend=30, severity=1):
    """ add  gamma noise """
    s = int(extend * abs(severity))
    n = np.random.gamma(shape=2, scale=s, size=im.shape)
    n = n - np.mean(n)
    im = cvt_uint8(np.float32(im) + n)
    return im


def noise_speckle(img, extend=40, severity=1):
    """ this creates larger 'blotches' of noise which look
    more realistic than just adding gaussian noise """
    severity = abs(severity) * extend
    blur = filters.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    return cvt_uint8(img + blur)


def noise_blur(im, severity=1):
    """add blur by shrinking an image and then enlarging to original size"""
    severity = abs(severity)
    f = 1 - 0.2 * severity
    h, w = im.shape[:2]
    hmin = 19.0
    f = max(f, hmin / h)
    im = cv2.resize(im, None, fx=f, fy=f)
    return np.array(cv2.resize(im, (w, h)))


def add_noise(img):
    """combine noises in np array"""
    img0 = img
    if lucky(0.1):
        img = noise_salt_pepper(img, uniform(0.3, 0.6))
    if lucky(0.2):
        img = noise_gaussian(img, uniform(0.3, 0.6))
    if lucky(0.5):
        img = noise_blur(img, uniform(0.3, 0.6))
    if lucky(0.5):
        img = noise_speckle(img, uniform(0.3, 0.6))
    if lucky(0.3):
        img = img // 2 + img0 // 2
    return img


def gaussian_blur(im, sigma=1, kernel_size=None, severity=1):
    """Gaussian blur, if kernel_size is passed in, severity will be invalid"""
    if kernel_size is None:
        step = 11
        kernel_size = int(step * severity)
        if kernel_size < 3.0:
            return im
        if kernel_size % 2 == 0:
            kernel_size -= 1
    return np.array(cv2.GaussianBlur(im, (kernel_size, kernel_size), sigma))


def rotate_shrink(im, max_angle=6, severity=0.5, cval=255):
    """rotate about center, shrink to keep the same size without cropping image"""
    max_angle = int(abs(severity) * max_angle)
    angle = randint(-max_angle, max_angle)
    h, w = im.shape[:2]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)
    nh = abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)
    scale = min(w / nw, h / nh)
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    im = cv2.warpAffine(im, mat, (w, h), borderValue=cval)
    return np.array(im)


def rotate_about_center(im, angle=4, scale=1, b_mode=None, cval=None, severity=1):
    """For the rotation effect, it is recommended to make b_mode not equal to None for color images, so that the
    filling will copy the edge pixel filling """
    angle = severity * angle
    if angle == 0:
        return im
    w = im.shape[1]
    h = im.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    if cval is None:
        cval = [0] * im.shape[2]
    elif isinstance(cval, (int, float)):
        cval = [cval] * im.shape[2]

    if b_mode is None:
        src = cv2.warpAffine(im, rot_mat, (int(ceil(nw)), int(ceil(nh))), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=cval)
    else:
        src = cv2.warpAffine(im, rot_mat, (int(ceil(nw)), int(ceil(nh))), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
    return np.array(src)


def randcrop(img, max_per=0.15, severity=1):
    """Random crop"""
    perc = max_per * severity
    rows, cols = img.shape[:2]
    k = int(rows * cols * perc / (rows + cols))
    roi = img[randint(0, k):rows - randint(0, k), randint(0, k):cols - randint(0, k)]
    return np.array(roi)


def enhance_sharpness(img, r=None, severity=1):
    """
    adjust the sharpness of an image. An
    enhancement factor of 0.0 gives a blurred image, a factor of 1.0 gives the
    original image, and a factor of 2.0 gives a sharpened image.
    """
    if r is None:
        severity = abs(severity)
        r = uniform(1 - 0.5 * severity, 1) if lucky(0.5) else uniform(1, 1 + severity)
    img = Image.fromarray(img)
    img = np.array(ImageEnhance.Sharpness(img).enhance(r))

    return img


def enhance_contrast(img, r=None, severity=1):
    """
    control the contrast of an image, similar
    to the contrast control on a TV set. An enhancement factor of 0.0
    gives a solid grey image. A factor of 1.0 gives the original image.
    """
    if r is None:
        severity = abs(severity)
        r = uniform(1 - 0.5 * severity, 1) if lucky(0.5) else uniform(1, 1 + severity)
    img = Image.fromarray(img)
    img = np.array(ImageEnhance.Contrast(img).enhance(r))

    return img


def enhance_brightness(img, r=None, severity=1):
    """
    control the brightness of an image.  An
    enhancement factor of 0.0 gives a black image. A factor of 1.0 gives the
    original image.
    """

    if r is None:
        severity = abs(severity)
        r = uniform(1 - 0.2 * severity, 1) if lucky(0.5) else uniform(1, 1 + severity * 0.5)
    img = Image.fromarray(img)
    img = np.array(ImageEnhance.Brightness(img).enhance(r))

    return img


def enhance_color(img, r=None, severity=1):
    """
    adjust the colour balance of an image, in
    a manner similar to the controls on a colour TV set. An enhancement
    factor of 0.0 gives a black and white image. A factor of 1.0 gives
    the original image.
    """
    if r is None:
        severity = abs(severity)
        r = uniform(1 - 0.5 * severity, 1) if lucky(0.5) else uniform(1, 1 + severity)

    img = Image.fromarray(img)
    img = np.array(ImageEnhance.Color(img).enhance(r))

    return img


def enhance(img):
    """combine image enhancement in the Image type, reduce conversions to np array"""
    if lucky(0.3):
        img = enhance_sharpness(img)
    if lucky(0.3):
        img = enhance_contrast(img)
    if lucky(0.3):
        img = enhance_brightness(img)
    return np.array(img)


def draw_line(im):
    """draw a line randomly"""
    h, w = im.shape[:2]
    p1 = (randint(0, w // 3), randint(0, h - 1))  # from left 1/3
    p2 = (randint(w // 3 * 2, w - 1), randint(0, h - 1))  # to right 1/3
    color = [randint(0, 255) for i in range(3)]
    lw = lucky_choice((1, 2), (0.8, 0.2))
    cv2.line(im, p1, p2, color, lw, cv2.LINE_AA)
    return np.array(im)


def center_im(im_outter, im_inner, shrink=True, vertical='center'):
    """center an image in a container image. `im_outter` can be the shape of it"""
    if not isinstance(im_outter, np.ndarray):
        shape = tuple(im_outter)
        if im_inner.ndim > len(shape):
            shape += im_inner.shape[len(shape):]
        im_outter = np.zeros(shape, np.uint8)

    H, W = im_outter.shape[:2]
    h, w = im_inner.shape[:2]
    if h > H or w > W:
        if shrink:
            rate = min(H / h, W / w)
            im_inner = cv2.resize(im_inner, rate)
        im_inner = im_inner[:H, :W]
        h, w = im_inner.shape[:2]

    vertical = vertical.lower()
    if vertical == 'center':
        dh = (H - h) // 2
    elif vertical == 'top':
        dh = 0
    elif vertical == 'bottom':
        dh = H - h

    im = im_outter.copy()
    dw = (W - w) // 2
    im[dh:dh + h, dw:dw + w] = im_inner
    return np.array(im)


def enhance_light(img):
    """combine image enhancement in the Image type, reduce conversions to np array"""
    if lucky(0.3):
        img = enhance_sharpness(img, uniform(0.5, 1.5))
    if lucky(0.3):
        img = enhance_contrast(img, uniform(0.7, 1.3))
    if lucky(0.3):
        img = enhance_brightness(img, uniform(0.85, 1.15))
    return np.array(img)


def gaussian2d(w, h):
    """The two-dimensional Gaussian distribution effect is actually an ellipse"""
    h = h // 2
    w = w // 2
    x = np.arange(-w, w)
    y = np.arange(-h, h)
    x, y = np.meshgrid(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    z = np.exp(
        -((y - mean_y) ** 2 / (std_y ** 2) + (x - mean_x) ** 2 / (std_x ** 2)) / 2
    )
    z /= (np.sqrt(2 * np.pi) * std_y)
    z *= 1 / (np.max(z) - np.min(z))
    return z


def add_stain(img, theta=200, severity=0.5, bright_spot=False, iteration=1):
    """Generate black stains or white bright spots"""

    for _ in range(0, iteration):
        img = np.float32(img)
        theta = theta * abs(severity)
        cols_big, rows_big = img.shape[:2]
        temp = min([cols_big, rows_big])

        if temp < 80:
            temp = 80
        if temp > 300:
            temp = 300

        if not bright_spot:
            gaussian_img = gaussian2d(randint(temp // 3, temp // 2), randint(temp // 3, temp // 2)) * theta
        else:
            gaussian_img = gaussian2d(randint(temp // 1.5, int(temp / 0.8)),
                                      randint(temp // 1.5, int(temp / 0.8)))

        cols_small, rows_small = gaussian_img.shape[:2]
        tmp_min = int(min(cols_small, rows_small))
        # 对椭圆效果做大幅度扭曲，cval最好不要过大。
        gaussian_img = rdistort(gaussian_img, randint(tmp_min // 10, tmp_min // 6), cval=0)
        x1 = randint(0, rows_big - 5 if rows_big - 5 > 0 else 0)
        y1 = randint(0, cols_big - 5 if cols_big - 5 > 0 else 0)

        if y1 + cols_small > cols_big:
            y2 = int(cols_big - 1)
        else:
            y2 = int(y1 + cols_small)

        if x1 + rows_small > rows_big:
            x2 = int(rows_big - 1)
        else:
            x2 = int(x1 + rows_small)

        row, col = gaussian_img.shape
        gaussian_img = gaussian_img.repeat(img.shape[2], axis=1)
        gaussian_img = gaussian_img.reshape(row, col, img.shape[2])

        gaussian_img = np.float32(gaussian_img[:(y2 - y1), :(x2 - x1)])
        if not bright_spot:
            img[y1:y2, x1:x2] -= gaussian_img
        else:
            temp1 = min([np.median(gaussian_img), 255 - np.mean(img[y1:y2, x1:x2])])
            gaussian_img = np.clip(gaussian_img - temp1, 0, 255)
            img[y1:y2, x1:x2] = np.clip(img[y1:y2, x1:x2] + gaussian_img, 0, 255)
        img = cvt_uint8(img)

    return np.array(img)


def shift_color(im, delta_max=10, severity=0.5):
    """randomly shift image color"""
    if severity == 0:
        return im

    delta_max = int(delta_max * severity)
    if isinstance(delta_max, tuple):
        delta_min, delta_max = delta_max
    else:
        delta_min = -delta_max

    im = np.float32(im)
    delta = np.random.randint(delta_min, delta_max, (1, 1, im.shape[2]))
    im += delta

    return np.array(cvt_uint8(im))


def random_contrast(img, contrast_delta=0.3, bright_delta=0.1):
    """randomly change image contrast and brightness"""
    if isinstance(contrast_delta, tuple):
        contrast_delta_min, contrast_delta = contrast_delta
    else:
        contrast_delta_min = -contrast_delta
    if isinstance(bright_delta, tuple):
        bright_delta_min, bright_delta = bright_delta
    else:
        bright_delta_min = -bright_delta
    fc = 1 + uniform(contrast_delta_min, contrast_delta)
    fb = 1 + uniform(bright_delta_min, bright_delta)
    im = img.astype(np.float32)
    if img.ndim == 2:
        im = im[:, :, None]
    mn = im.mean(axis=(0, 1), keepdims=True)
    im = (im - mn) * fc + mn * fb
    im = im.clip(0, 255).astype(np.uint8)
    return np.array(im)


def period_map(xi, times, extent):
    if times < 1:
        return None
    times = float(times)
    theta = randint(extent, extent + 10) * choice([1, -1])

    def back(x):
        if x < times / 2.0:
            # Here only the effect of a sin function is achieved, and more effects can be added later.
            return theta * sin(pi * (3 / 2.0 + x / times))  # Monotonically increasing
        return theta * sin(pi * (1 / 2.0 + x / times))

    xi = np.fabs(xi)
    xi = xi % times
    yi = np.array(list(map(back, xi)))
    return yi


def whole_rdistort(im, severity=1, scop=40):
    """
    Using the affine projection method in skimg,
    Realize the picture through the corresponding coordinate projection
    Specifies the distortion effect of the form. This function will normalize 0-1
    """

    if severity == 0:
        return im

    theta = severity * scop
    rows, cols = im.shape[:2]
    colpoints = max(int(cols * severity * 0.05), 3)
    rowpoints = max(int(rows * severity * 0.05), 3)

    src_cols = np.linspace(0, cols, colpoints)
    src_rows = np.linspace(0, rows, rowpoints)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # The key location for wave distortion effect
    dst_rows = src[:, 1] - period_map(np.linspace(0, 100, src.shape[0]), 50, 20)

    # dst columns
    dst_cols = src[:, 0] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * theta

    dst = np.vstack([dst_cols, dst_rows]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    image = warp(im, tform, mode='edge', output_shape=(rows, cols)) * 255
    return np.array(cvt_uint8(image))


def lucky_choice(seq, ps=None, rand_func=random):
    """randomly choose an element from `seq` according to their probability distribution `ps`"""
    if not seq:
        return None
    if ps is None:
        return choice(seq)
    cumps = np.cumsum(ps)
    r = rand_func() * cumps[-1]
    idx = (cumps < r).sum()
    idx = min(idx, len(seq) - 1)
    return seq[idx]


def cvt_uint8(im):
    """convert image type to `np.uint8`"""
    if im.dtype == np.uint8:
        return im
    return np.round(im).clip(0, 255).astype(np.uint8)


def to_image(im):
    """convert `im` to `Image` type"""
    if not isinstance(im, Image.Image):
        if im.ndim == 3:
            im = im[:, :, ::-1]  # reverse channels: BGR in cv2 to RGB in Image
        im = Image.fromarray(im)
    return im


def to_array(im):
    """convert `im` to `np.array` type"""
    if isinstance(im, Image.Image):
        im = np.array(im)
        if im.ndim == 3:
            im = im[:, :, ::-1]  # reverse channels: RGB in Image to BGR in cv2
    return im


def unify_img(img, img_height=64, max_length=512, img_channel=3):
    color_fill = 255
    img_shape = img.shape

    img_width = int(float(img_shape[1]) / img_shape[0] * img_height)
    img = cv2.resize(img, (img_width, img_height))
    if img_width > max_length:
        img = img[:, 0:max_length]
    else:
        blank_img = np.zeros((img_height, max_length, img_channel), np.uint8)
        # fill the image with white
        blank_img.fill(color_fill)
        blank_img[0:img_height, 0:img_width] = img
        img = blank_img
    return np.array(img)


def unify_img_label(img, label, img_height=64, max_length=512, min_length=192, img_channel=3):
    color_fill = 255
    img_shape = img.shape

    img_width = int(float(img_shape[1]) / img_shape[0] * img_height)
    img = cv2.resize(img, (img_width, img_height))
    if img_width > max_length:
        img = img[:, 0:max_length]
    else:
        blank_img = np.zeros((img_height, max_length, img_channel), np.uint8)
        # fill the image with white
        blank_img.fill(color_fill)
        blank_img[0:img_height, 0:img_width] = img
        img = blank_img

    return np.array(img), label
