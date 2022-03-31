mindspore.dataset.vision.c_transforms.Decode
============================================

.. py:class:: mindspore.dataset.vision.c_transforms.Decode(rgb=True)

    对输入图像进行解码。

    **参数：**

    - **rgb**  (bool，可选) - 解码输入图像的模式。若为True，将输入解码为RGB图像；否则为BGR图像(已弃用)。默认值：True。

    **异常：**

    - **RuntimeError** - 如果 `rgb` 为 False，因为此选项已弃用。
    - **RuntimeError** - 如果输入图像不是一维序列。
