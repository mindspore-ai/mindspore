mindspore.dataset.vision.py_transforms.ToTensor
===============================================

.. py:class:: mindspore.dataset.vision.py_transforms.ToTensor(output_type=numpy.float32)

    将输入的PIL或numpy.ndarray图像转换为指定数据类型的numpy.ndarray图像，此时像素值取值将由[0, 255]变为[0.0, 1.0]，图像的shape将由(H, W, C)变为(C, H, W)。

    .. note:: 输入图像中的像素值将从 [0, 255] 缩放为 [0.0, 1.0]。图像的数据类型将被转换为由 `output_type` 参数指定的类型，图像通道数保持不变。

    **参数：**

    - **output_type** (numpy.dtype，可选) - 输出图像的数据类型，默认值：numpy.float32。

    **异常：**

    - **TypeError** - 当输入图像的类型不为 :class:`numpy.ndarray` 或 :class:`PIL.Image` 。
    - **TypeError** - 输入图像的维度不为2或3。
