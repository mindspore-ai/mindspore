# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""
This module is to support vision augmentations.
Some image augmentations are implemented with C++ OpenCV to provide high performance.
Other additional image augmentations are developed with Python PIL.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore.dataset.vision.utils as utils

Note: Legacy c_transforms and py_transforms are deprecated but can still be imported as follows:

.. code-block::

    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.vision.py_transforms as py_vision

See `Vision Transforms
<https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html#vision-transforms>`_ tutorial for more details.

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- ImageTensorOperation, the base class of all image processing operations. It is a derived class of TensorOperation.
- PyTensorOperation, the base class of all data processing operations implemented in Python.

The data transform operation can be executed in the data processing pipeline or in the eager mode:

- Pipeline mode is generally used to process datasets. For examples, please refer to
  `introduction to data processing pipeline <https://www.mindspore.cn/docs/en/master/api_python/
  mindspore.dataset.html#introduction-to-data-processing-pipeline>`_ .
- Eager mode is generally used for scattered samples. Examples of image preprocessing are as follows:

  .. code-block::

      import numpy as np
      import mindspore.dataset.vision as vision
      from PIL import Image,ImageFont,ImageDraw

      # draw circle
      img = Image.new("RGB", (300, 300), (255, 255, 255))
      draw = ImageDraw.Draw(img)
      draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
      img.save("./1.jpg")
      with open("./1.jpg", "rb") as f:
          data = f.read()

      data_decoded = vision.Decode()(data)
      data_croped = vision.RandomCrop(size=(250, 250))(data_decoded)
      data_resized = vision.Resize(size=(224, 224))(data_croped)
      data_normalized = vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                         std=[0.229 * 255, 0.224 * 255, 0.225 * 255])(data_resized)
      data_hwc2chw = vision.HWC2CHW()(data_normalized)
      print("data: {}, shape: {}".format(data_hwc2chw, data_hwc2chw.shape), flush=True)
"""
from . import c_transforms
from . import py_transforms
from . import transforms
from . import utils
from .transforms import AdjustBrightness, AdjustContrast, AdjustGamma, AdjustHue, AdjustSaturation, AdjustSharpness, \
    Affine, AutoAugment, AutoContrast, BoundingBoxAugment, CenterCrop, ConvertColor, Crop, CutMixBatch, CutOut, \
    Decode, Equalize, Erase, FiveCrop, GaussianBlur, Grayscale, HorizontalFlip, HsvToRgb, HWC2CHW, Invert, \
    LinearTransformation, MixUp, MixUpBatch, Normalize, NormalizePad, Pad, PadToSize, Perspective, Posterize, \
    RandAugment, RandomAdjustSharpness, RandomAffine, RandomAutoContrast, RandomColor, RandomColorAdjust, RandomCrop, \
    RandomCropDecodeResize, RandomCropWithBBox, RandomEqualize, RandomErasing, RandomGrayscale, RandomHorizontalFlip, \
    RandomHorizontalFlipWithBBox, RandomInvert, RandomLighting, RandomPerspective, RandomPosterize, RandomResizedCrop, \
    RandomResizedCropWithBBox, RandomResize, RandomResizeWithBBox, RandomRotation, RandomSelectSubpolicy, \
    RandomSharpness, RandomSolarize, RandomVerticalFlip, RandomVerticalFlipWithBBox, Rescale, Resize, ResizedCrop, \
    ResizeWithBBox, RgbToHsv, Rotate, SlicePatches, Solarize, TenCrop, ToNumpy, ToPIL, ToTensor, ToType, \
    TrivialAugmentWide, UniformAugment, VerticalFlip, not_random
from .utils import AutoAugmentPolicy, Border, ConvertMode, ImageBatchFormat, ImageReadMode, Inter, SliceMode, \
    encode_jpeg, encode_png, get_image_num_channels, get_image_size, read_file, read_image, write_file, write_jpeg, \
    write_png
