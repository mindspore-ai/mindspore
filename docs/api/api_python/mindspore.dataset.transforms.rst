mindspore.dataset.transforms
============================

通用
-----

此模块用于通用数据增强，其中一部分增强操作是用C++实现的，具有较好的高性能，另一部分是基于Python实现，使用了NumPy模块作为支持。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- PyTensorOperation，所有Python实现的数据处理操作的基类。

变换
^^^^^

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.TypeCast

工具
^^^^^

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst


视觉
-----

此模块用于图像数据增强，其中有一部分增强是基于C++ OpenCV实现的，具有较好的性能，而另一部分是基于Python Pillow实现的。

API样例中常用的导入模块如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore.dataset.vision.utils as utils

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- ImageTensorOperation，所有图像类数据处理操作的基类，继承自TensorOperation。
- PyTensorOperation，所有Python实现的数据处理操作的基类。

数据增强操作可以放入数据处理Pipeline中执行，也可以Eager模式执行：

- Pipeline模式用于流式处理大型数据集，示例可参考 `数据处理Pipeline介绍 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.loading.html#数据处理pipeline介绍>`_ 。
- Eager模式用于函数调用方式处理样本，示例可参考 `轻量化数据处理 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/eager.html>`_ 。

样例库
^^^^^^

快速上手使用视觉类变换的API，跳转参考 `视觉变换样例库 <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_ 。
此指南中展示了多个变换API的用法，以及输入输出结果。

变换
^^^^^

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.CenterCrop
    mindspore.dataset.vision.Decode
    mindspore.dataset.vision.HWC2CHW
    mindspore.dataset.vision.Normalize
    mindspore.dataset.vision.RandomColorAdjust
    mindspore.dataset.vision.RandomCrop
    mindspore.dataset.vision.RandomCropDecodeResize
    mindspore.dataset.vision.RandomHorizontalFlip
    mindspore.dataset.vision.RandomResizedCrop
    mindspore.dataset.vision.RandomVerticalFlip
    mindspore.dataset.vision.Rescale
    mindspore.dataset.vision.Resize
    mindspore.dataset.vision.ToNumpy
    mindspore.dataset.vision.ToPIL
    mindspore.dataset.vision.ToTensor

工具
^^^^^

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.ImageReadMode
    mindspore.dataset.vision.Border
    mindspore.dataset.vision.ImageBatchFormat
    mindspore.dataset.vision.Inter

文本处理接口说明
----------------

``mindspore.dataset.text`` 及其词表、分词、文本变换接口已移除。
数据集的词表构建方法和含上述变换的旧序列化管线也需要迁移。
文本数据集读取、通用 ``Dataset.map``、MindRecord 和 DataLoader 仍可使用。
可在数据预处理阶段完成分词并保存 token IDs，或通过 ``Dataset.map`` 调用自行选择的 Python 文本处理函数。
替换时应核对词表和特殊 token、大小写与 Unicode 规范化、分词规则、偏移量单位、shape/dtype 及吞吐。
不同分词器和外部库不保证直接等价，词表训练与模型文件的兼容性也需单独验收。
