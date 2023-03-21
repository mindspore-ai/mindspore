mindspore.dataset.vision.ToTensor
=================================

.. py:class:: mindspore.dataset.vision.ToTensor(output_type=np.float32)

    将输入PIL图像或numpy.ndarray图像转换为指定类型的numpy.ndarray图像，图像的像素值范围将从[0, 255]放缩为[0.0, 1.0]，shape将从<H, W, C>调整为<C, H, W>。

    参数：
        - **output_type** (Union[mindspore.dtype, numpy.dtype]，可选) - 输出图像的数据类型。默认值：`numpy.float32` 。

    异常：
        - **TypeError** - 当输入图像的类型不为 `PIL.Image.Image` 或 :class:`numpy.ndarray` 。
        - **TypeError** - 输入图像的维度不为2或3。
