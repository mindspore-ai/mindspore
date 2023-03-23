mindspore.dataset.vision.RandomPosterize
========================================

.. py:class:: mindspore.dataset.vision.RandomPosterize(bits=(8, 8))

    随机减少图像的颜色通道的比特位数，使图像变得高对比度和颜色鲜艳。

    参数：
        - **bits** (Union[int, Sequence[int]], 可选) - 随机位数压缩的范围。位值必须在 [1,8] 范围内，并且在给定范围内至少包含一个整数值。它必须是 (min, max) 或整数格式。
          如果min与max相等，那么它是一个单一的位数压缩操作。默认值：(8, 8)。

    异常：
        - **TypeError** - 如果 `bits` 不是int或Sequence[int]类型。
        - **ValueError** - 如果 `bits` 不在 [1, 8] 范围内。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
