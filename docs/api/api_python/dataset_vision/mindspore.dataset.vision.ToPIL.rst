mindspore.dataset.vision.ToPIL
==============================

.. py:class:: mindspore.dataset.vision.ToPIL

    将已解码的numpy.ndarray图像转换为PIL图像。

    .. note:: 转换模式将根据 `PIL.Image.fromarray` 由图像的数据类型决定。

    异常：
        - **TypeError** - 当输入图像的类型不为 :class:`numpy.ndarray` 或 `PIL.Image.Image` 。
