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
"""Data transform."""
import random
import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    print('OpenCV is not installed and cannot be used')
    cv2 = None
__all__ = ['RandomCropLetterbox', 'RandomFlip', 'HSVShift', 'ResizeLetterbox']


class RandomCropLetterbox():
    """ Take random crop from the image.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(number or list, optional): Argument passed on to :class:
        `brambox.boxes.util.modifiers.CropModifier`
    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """

    def __init__(self, jitter, fill_color=127, input_dim=(1408, 768)):
        self.fill_color = fill_color
        self.jitter = jitter
        self.crop_info = None
        self.output_w = None
        self.output_h = None
        self.input_dim = input_dim

    def __call__(self, img, annos):
        if img is None:
            return None, None
        if isinstance(img, Image.Image):
            img, _ = self._tf_pil(img)
        elif isinstance(img, np.ndarray):
            img, _ = self._tf_cv(img)
        annos = self._tf_anno(annos)
        annos = np.asarray(annos)

        return (img, annos)

    def _tf_cv(self, img, save_info=None):
        """ Take random crop from image """
        self.output_w, self.output_h = self.input_dim
        orig_h, orig_w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        dw = int(self.jitter * orig_w)
        dh = int(self.jitter * orig_h)

        if save_info is None:
            new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
        else:
            new_ar = save_info[0]

        if save_info is None:
            scale = random.random() * (2 - 0.25) + 0.25
        else:
            scale = save_info[1]
        if new_ar < 1:
            nh = int(scale * orig_h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * orig_w)
            nh = int(nw / new_ar)

        if save_info is None:
            if self.output_w > nw:
                dx = random.randint(0, self.output_w - nw)
            else:
                dx = random.randint(self.output_w - nw, 0)
        else:
            dx = save_info[2]

        if save_info is None:
            if self.output_h > nh:
                dy = random.randint(0, self.output_h - nh)
            else:
                dy = random.randint(self.output_h - nh, 0)
        else:
            dy = save_info[3]

        nxmin = max(0, -dx)
        nymin = max(0, -dy)
        nxmax = min(nw, -dx + self.output_w - 1)
        nymax = min(nh, -dy + self.output_h - 1)
        sx, sy = float(orig_w) / nw, float(orig_h) / nh
        orig_xmin = int(nxmin * sx)
        orig_ymin = int(nymin * sy)
        orig_xmax = int(nxmax * sx)
        orig_ymax = int(nymax * sy)

        orig_crop = img[orig_ymin:orig_ymax, orig_xmin:orig_xmax, :]
        orig_crop_resize = cv2.resize(orig_crop, (nxmax - nxmin, nymax - nymin), interpolation=cv2.INTER_CUBIC)

        output_img = np.ones((self.output_h, self.output_w, channels), dtype=np.uint8) * self.fill_color

        y_lim = int(min(output_img.shape[0], orig_crop_resize.shape[0]))
        x_lim = int(min(output_img.shape[1], orig_crop_resize.shape[1]))

        output_img[:y_lim, :x_lim, :] = orig_crop_resize[:y_lim, :x_lim, :]
        self.crop_info = [sx, sy, nxmin, nymin, nxmax, nymax]
        if save_info is None:
            return output_img, [new_ar, scale, dx, dy]

        return output_img, save_info

    def _tf_pil(self, img, save_info=None):
        """ Take random crop from image """
        self.output_w, self.output_h = self.input_dim
        orig_w, orig_h = img.size
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        dw = int(self.jitter * orig_w)
        dh = int(self.jitter * orig_h)
        if save_info is None:
            new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
        else:
            new_ar = save_info[0]

        if save_info is None:
            scale = random.random() * (2 - 0.25) + 0.25

        else:
            scale = save_info[1]
        if new_ar < 1:
            nh = int(scale * orig_h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * orig_w)
            nh = int(nw / new_ar)

        if save_info is None:
            if self.output_w > nw:
                dx = random.randint(0, self.output_w - nw)
            else:
                dx = random.randint(self.output_w - nw, 0)
        else:
            dx = save_info[2]

        if save_info is None:
            if self.output_h > nh:
                dy = random.randint(0, self.output_h - nh)
            else:
                dy = random.randint(self.output_h - nh, 0)
        else:
            dy = save_info[3]

        nxmin = max(0, -dx)
        nymin = max(0, -dy)
        nxmax = min(nw, -dx + self.output_w - 1)
        nymax = min(nh, -dy + self.output_h - 1)
        sx, sy = float(orig_w) / nw, float(orig_h) / nh
        orig_xmin = int(nxmin * sx)
        orig_ymin = int(nymin * sy)
        orig_xmax = int(nxmax * sx)
        orig_ymax = int(nymax * sy)
        orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
        orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
        output_img = Image.new(img.mode, (self.output_w, self.output_h), color=(self.fill_color,) * channels)
        output_img.paste(orig_crop_resize, (0, 0))
        self.crop_info = [sx, sy, nxmin, nymin, nxmax, nymax]
        if save_info is None:
            return output_img, [new_ar, scale, dx, dy]

        return output_img, save_info

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous crop """
        def is_negative(anno):
            for value in anno:
                if value != -1:
                    return False
            return True
        sx, sy, crop_xmin, crop_ymin, crop_xmax, crop_ymax = self.crop_info
        for i in range(len(annos) - 1, -1, -1):
            anno = annos[i]
            if is_negative(anno):
                continue
            else:
                x1 = max(crop_xmin, int(anno[1] / sx))
                x2 = min(crop_xmax, int((anno[1] + anno[3]) / sx))
                y1 = max(crop_ymin, int(anno[2] / sy))
                y2 = min(crop_ymax, int((anno[2] + anno[4]) / sy))
                w = x2 - x1
                h = y2 - y1

                if w <= 2 or h <= 2:
                    annos[i] = np.zeros(6)
                    continue

                annos[i][1] = x1 - crop_xmin
                annos[i][2] = y1 - crop_ymin
                annos[i][3] = w
                annos[i][4] = h
        return annos


class RandomFlip():
    """ Randomly flip image.

    Args:
        threshold (Number [0-1]): Chance of flipping the image

    Note:
        Create 1 RandomFlip object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.flip = False
        self.im_w = None

    def __call__(self, img, annos):
        if img is None and annos is None:
            return None, None
        if isinstance(img, Image.Image):
            img = self._tf_pil(img)
        elif isinstance(img, np.ndarray):
            img = self._tf_cv(img)
        annos = [self._tf_anno(anno) for anno in annos]
        annos = np.asarray(annos)

        return (img, annos)

    def _tf_pil(self, img):
        """ Randomly flip image """
        self.flip = self._get_flip()
        self.im_w = img.size[0]

        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _tf_cv(self, img):
        """ Randomly flip image """
        self.flip = self._get_flip()
        self.im_w = img.shape[1]

        if self.flip:
            img = cv2.flip(img, 1)
        return img

    def _get_flip(self):
        flip = random.random() < self.threshold
        return flip

    def _tf_anno(self, anno):
        """ Change coordinates of an annotation, according to the previous flip """
        def is_negative(anno):
            for value in anno:
                if value not in (-1, 0):
                    return False
            return True
        if is_negative(anno):
            return anno
        if self.flip and self.im_w is not None:
            anno[1] = self.im_w - anno[1] - anno[3]
        return anno


class HSVShift():
    """ Perform random HSV shift on the RGB data.

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue
        saturation (Number): Random number between 1,saturation is used to shift the saturation; 50% chance to
        get 1/dSaturation in stead of dSaturation
        value (Number): Random number between 1,value is used to shift the value; 50% chance to get 1/dValue in
        stead of dValue
    Warning:
        If you use OpenCV as your image processing library, make sure the image is RGB before using this transform.
        By default OpenCV uses BGR, so you must use `cvtColor`_ function to transform it to RGB.
    .. _cvtColor: https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
    """

    def __init__(self, hue, saturation, value):
        self.hue = hue
        self.saturation = saturation
        self.value = value

    def __call__(self, img, annos):
        dh = random.uniform(-self.hue, self.hue)
        ds = random.uniform(1, self.saturation)
        if random.random() < 0.5:
            ds = 1 / ds
        dv = random.uniform(1, self.value)
        if random.random() < 0.5:
            dv = 1 / dv

        if img is None:
            return None
        if isinstance(img, Image.Image):
            img = self._tf_pil(img, dh, ds, dv)
            return (img, annos)
        if isinstance(img, np.ndarray):
            return (self._tf_cv(img, dh, ds, dv), annos)

        print(f'HSVShift only works with <PIL images> or <OpenCV images> [{type(img)}]')
        return (img, annos)

    @staticmethod
    def _tf_pil(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.convert('HSV')
        channels = list(img.split())

        def change_hue(x):
            x += int(dh * 255)
            if x > 255:
                x -= 255
            elif x < 0:
                x += 0
            return x

        channels[0] = channels[0].point(change_hue)
        channels[1] = channels[1].point(lambda i: min(255, max(0, int(i * ds))))
        channels[2] = channels[2].point(lambda i: min(255, max(0, int(i * dv))))

        img = Image.merge(img.mode, tuple(channels))
        img = img.convert('RGB')
        return img

    @staticmethod
    def _tf_cv(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        def wrap_hue(x):
            x[x >= 360.0] -= 360.0
            x[x < 0.0] += 360.0
            return x

        img[:, :, 0] = wrap_hue(img[:, :, 0] + (360.0 * dh))
        img[:, :, 1] = np.clip(ds * img[:, :, 1], 0.0, 1.0)
        img[:, :, 2] = np.clip(dv * img[:, :, 2], 0.0, 1.0)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = (img * 255).astype(np.uint8)
        return img


class ResizeLetterbox:
    """ Resize the image to input_dim.

    Args:
        input_dim: Input size of network.
    """

    def __init__(self, fill_color=127, input_dim=(1408, 768)):
        self.fill_color = fill_color
        self.crop_info = None
        self.output_w = None
        self.output_h = None
        self.input_dim = input_dim
        self.pad = None
        self.scale = None

    def __call__(self, img, annos):
        if img is None:
            return None, None
        if isinstance(img, Image.Image):
            img = self._tf_pil(img)

        annos = np.asarray(annos)

        return img, annos

    def _tf_pil(self, img):
        """ Letterbox an image to fit in the network """

        net_w, net_h = self.input_dim

        im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            resample_mode = Image.NEAREST
            img = img.resize((int(self.scale * im_w), int(self.scale * im_h)), resample_mode)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w + .5), int(pad_h + .5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,) * channels)

        return img
