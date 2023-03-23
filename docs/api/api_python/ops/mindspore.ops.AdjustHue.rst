mindspore.ops.AdjustHue
=======================

.. py:class:: mindspore.ops.AdjustHue

    调整 RGB 图像的色调。

    .. note::
        该运算是种将 RGB 图像转换为浮点表示的便捷方法。通过将图像转换为HSV色彩空间并移动色调通道中的强度来调整图像，然后转换回原始数据模式。
        当多个调整依次进行时尽量减少冗余转换的数量。

    输入：
        - **image** (Tensor) - RGB格式的图片Tensor，维度至少为三。最后一个维度表示通道数，大小必须为3。dtype为float16或float32。
        - **delta** (Tensor) - 色调通道的添加值。dtype需要是float32。Tensor必须是零维的。

    输出：
        Tensor，具有与 `image` 相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `image` 或 `delta` 不是Tensor。
        - **TypeError** - 如果 `image` 的dtype不是：float32或float16。
        - **TypeError** - 如果 `delta` 的dtype不是：float32。
        - **ValueError** - 如果 `image` 的维度低于三维。
