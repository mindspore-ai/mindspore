mindspore.dataset.vision
===================================

此模块用于图像数据增强，其中有一部分增强是基于C++ OpenCV实现的，具有较好的性能，而另一部分是基于Python Pillow实现的。

API样例中常用的导入模块如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore.dataset.vision.utils as utils

注意：旧的API导入方式已经过时且会逐步废弃，因此推荐使用上面的方式，但目前仍可按以下方式导入：

.. code-block::

    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.vision.py_transforms as py_vision
    from mindspore.dataset.transforms import c_transforms

更多详情请参考 `图像数据加载与增强 <https://www.mindspore.cn/tutorials/zh-CN/r1.9/advanced/dataset/augment_image_data.html>`_。

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- PyTensorOperation，所有Python实现的数据处理操作的基类。

数据增强算子可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式一般用于处理数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore.dataset.html#数据处理pipeline介绍>`_。
- Eager模式一般用于零散样本，图像预处理举例如下：

  .. code-block::

      import numpy as np
      import mindspore.dataset.vision as vision
      from PIL import Image,ImageFont,ImageDraw

      # 画圆形
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

变换
-----

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.AdjustGamma
    mindspore.dataset.vision.AutoAugment
    mindspore.dataset.vision.AutoContrast
    mindspore.dataset.vision.BoundingBoxAugment
    mindspore.dataset.vision.CenterCrop
    mindspore.dataset.vision.ConvertColor
    mindspore.dataset.vision.Crop
    mindspore.dataset.vision.CutMixBatch
    mindspore.dataset.vision.CutOut
    mindspore.dataset.vision.Decode
    mindspore.dataset.vision.Equalize
    mindspore.dataset.vision.FiveCrop
    mindspore.dataset.vision.GaussianBlur
    mindspore.dataset.vision.Grayscale
    mindspore.dataset.vision.HorizontalFlip
    mindspore.dataset.vision.HsvToRgb
    mindspore.dataset.vision.HWC2CHW
    mindspore.dataset.vision.Invert
    mindspore.dataset.vision.LinearTransformation
    mindspore.dataset.vision.MixUp
    mindspore.dataset.vision.MixUpBatch
    mindspore.dataset.vision.Normalize
    mindspore.dataset.vision.NormalizePad
    mindspore.dataset.vision.Pad
    mindspore.dataset.vision.PadToSize
    mindspore.dataset.vision.RandomAdjustSharpness
    mindspore.dataset.vision.RandomAffine
    mindspore.dataset.vision.RandomAutoContrast
    mindspore.dataset.vision.RandomColor
    mindspore.dataset.vision.RandomColorAdjust
    mindspore.dataset.vision.RandomCrop
    mindspore.dataset.vision.RandomCropDecodeResize
    mindspore.dataset.vision.RandomCropWithBBox
    mindspore.dataset.vision.RandomEqualize
    mindspore.dataset.vision.RandomErasing
    mindspore.dataset.vision.RandomGrayscale
    mindspore.dataset.vision.RandomHorizontalFlip
    mindspore.dataset.vision.RandomHorizontalFlipWithBBox
    mindspore.dataset.vision.RandomInvert
    mindspore.dataset.vision.RandomLighting
    mindspore.dataset.vision.RandomPerspective
    mindspore.dataset.vision.RandomPosterize
    mindspore.dataset.vision.RandomResizedCrop
    mindspore.dataset.vision.RandomResizedCropWithBBox
    mindspore.dataset.vision.RandomResize
    mindspore.dataset.vision.RandomResizeWithBBox
    mindspore.dataset.vision.RandomRotation
    mindspore.dataset.vision.RandomSelectSubpolicy
    mindspore.dataset.vision.RandomSharpness
    mindspore.dataset.vision.RandomSolarize
    mindspore.dataset.vision.RandomVerticalFlip
    mindspore.dataset.vision.RandomVerticalFlipWithBBox
    mindspore.dataset.vision.Rescale
    mindspore.dataset.vision.Resize
    mindspore.dataset.vision.ResizeWithBBox
    mindspore.dataset.vision.RgbToHsv
    mindspore.dataset.vision.Rotate
    mindspore.dataset.vision.SlicePatches
    mindspore.dataset.vision.TenCrop
    mindspore.dataset.vision.ToNumpy
    mindspore.dataset.vision.ToPIL
    mindspore.dataset.vision.ToTensor
    mindspore.dataset.vision.ToType
    mindspore.dataset.vision.UniformAugment
    mindspore.dataset.vision.VerticalFlip

工具
-----

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.AutoAugmentPolicy
    mindspore.dataset.vision.Border
    mindspore.dataset.vision.ConvertMode
    mindspore.dataset.vision.ImageBatchFormat
    mindspore.dataset.vision.Inter
    mindspore.dataset.vision.SliceMode
    mindspore.dataset.vision.get_image_num_channels
    mindspore.dataset.vision.get_image_size
