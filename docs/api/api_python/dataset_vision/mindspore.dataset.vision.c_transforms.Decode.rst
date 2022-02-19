mindspore.dataset.vision.c_transforms.Decode
============================================

.. py:class:: mindspore.dataset.vision.c_transforms.Decode(rgb=True)

    以 RGB 模式（默认）或 BGR 模式（选项已弃用）解码输入图像。

    **参数：**

    - **rgb**  (bool，可选) - 解码输入图像的模式, 默认值：True。
      如果 True 表示解码图像的格式为 RGB，否则为 BGR（选项已弃用）。

    **异常：**

    - **RuntimeError** - 如果 `rgb` 为 False，因为此选项已弃用。
    - **RuntimeError** - 如果输入图像的shape不是一维序列。
