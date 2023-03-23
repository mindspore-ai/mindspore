mindspore.ops.AdjustSaturation
==============================

.. py:class:: mindspore.ops.AdjustSaturation

    调整 RGB 图像的饱和度。

    .. note::
        该运算是种将 RGB 图像转换为浮点表示的便捷方法。通过将图像转换为HSV色彩空间并调整饱和度通道的强度来调整图像，然后转换回原始数据模式。
        当多个调整依次进行时尽量减少冗余转换的数量。

    输入：
        - **image** (Tensor) - 输入的Tensor。dtype需要是float16或float32。Tensor的维度至少是三维。shape的最后一个维度表示通道数，且必须为3。
        - **scale** (Tensor) - 应用于图像的饱和度调整因子。大于1.0的值增加饱和度，小于1.0的值减少饱和度，等于1.0保持饱和度不变。Tensor必须是零维，其dtype必须是float32。

    输出：
        Tensor，具有与 `image` 相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `image` 或 `scale` 不是Tensor。
        - **TypeError** - 如果 `image` 的dtype不是：float32或float16。
        - **TypeError** - 如果 `scale` 的dtype不是：float32。
        - **ValueError** - 如果 `image` 的维度低于三维。
        - **ValueError** - 如果 `image` 的shape的最后一个维度不是3。
