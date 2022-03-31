mindspore.dataset.vision.py_transforms.HWC2CHW
==============================================

.. py:class:: mindspore.dataset.vision.py_transforms.HWC2CHW

    将输入的numpy.ndarray图像的shape从(H, W, C)转换为(C, H, W)。

    **异常：**

    - **TypeError** - 当输入图像的类型不为 :class:`numpy.ndarray` 。
    - **TypeError** - 当输入图像的维度不为3。
