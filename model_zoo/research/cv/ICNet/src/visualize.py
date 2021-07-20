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
"""visualize segmentation"""
import os
import sys
import numpy as np
from PIL import Image
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import load_param_into_net
from mindspore import load_checkpoint
import mindspore.dataset.vision.py_transforms as transforms
from models.icnet import ICNet

__all__ = ['get_color_palette', 'set_img_color',
           'show_prediction', 'show_colorful_images', 'save_colorful_images']


def _img_transform(img):
    """img_transform"""
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    img = totensor(img)
    img = normalize(img)
    return img


def set_img_color(img, label, colors, background=0, show255=False):
    for i in enumerate(colors):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pre, colors, background=0):
    im = np.array(img, np.uint8)
    pre = pre
    set_img_color(im, pre, colors, background)
    out = np.array(im)

    return out


def show_colorful_images(prediction, palettes):
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    im.show()


def save_colorful_images(prediction, filename, output_dir, palettes):
    """param prediction: [B, H, W, C]"""
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


def get_color_palette(npimg, dataset='pascal_voc'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color palette
    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255
    # put colormap
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(vocpalette)
    return out_img


def _getvocpalette(num_cls):
    """get_vocpalette"""
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return palette


vocpalette = _getvocpalette(256)


def _class_to_index(mask):
    """assert the value"""
    values = np.unique(mask)
    _key = np.array([-1, -1, -1, -1, -1, -1,
                     -1, -1, 0, 1, -1, -1,
                     2, 3, 4, -1, -1, -1,
                     5, -1, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15,
                     -1, -1, 16, 17, 18])
    _mapping = np.array(range(-1, len(_key) - 1)).astype('int32')
    for value in values:
        assert value in _mapping
    # Get the index of each pixel value in the mask corresponding to _mapping
    index = np.digitize(mask.ravel(), _mapping, right=True)
    # According to the above index index, according to _key, the corresponding mask image is obtained
    return _key[index].reshape(mask.shape)


def _mask_transform(mask):
    mask = _class_to_index(np.array(mask).astype('int32'))
    return np.array(mask).astype('int32')


if __name__ == '__main__':
    sys.path.append('/root/ICNet/src/')
    model = ICNet(nclass=19, backbone='resnet50', istraining=False)
    ckpt_file_name = '/root/ICNet/ckpt/ICNet-160_93_699.ckpt'
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(model, param_dict)
    image_path = '../Test/val_lindau_000023_000019_leftImg8bit.png'
    image = Image.open(image_path).convert('RGB')
    image = _img_transform(image)
    image = Tensor(image)

    expand_dims = ops.ExpandDims()
    image = expand_dims(image, 0)

    squeeze = ops.Squeeze()
    outputs = model(image)
    pred = ops.Argmax(axis=1)(outputs[0])
    pred = pred.asnumpy()
    pred = pred.squeeze(0)
    pred = get_color_palette(pred, "citys")
    pred.save('Test/visual_pred_random.png')
