mindspore.dataset.vision.py_transforms.ToType
=============================================

.. py:class:: mindspore.dataset.vision.py_transforms.ToType(output_type)

    将输入的numpy.ndarray图像转换为指定数据类型。

    **参数：**

    - **output_type** (numpy.dtype) - 输出图像的数据类型，例如numpy.float32。

    **异常：**

    - **TypeError** - 输入图像的类型不为 :class:`numpy.ndarray` 。
