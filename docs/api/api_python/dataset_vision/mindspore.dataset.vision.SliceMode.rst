mindspore.dataset.vision.SliceMode
==================================

.. py:class:: mindspore.dataset.vision.SliceMode

    Tensor切片方式枚举类。

    可选枚举值为： ``SliceMode.PAD`` 、 ``SliceMode.DROP``。

    - **SliceMode.PAD** - 当图像无法进行整数块切分时，填充最后一个图像块至指定切分尺寸大小。
    - **SliceMode.DROP** - 当图像无法进行整数块切分时，丢弃最后一个图像块。
