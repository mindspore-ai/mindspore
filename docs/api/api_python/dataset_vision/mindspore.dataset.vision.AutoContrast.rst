mindspore.dataset.vision.AutoContrast
=====================================

.. py:class:: mindspore.dataset.vision.AutoContrast(cutoff=0.0, ignore=None)

    在输入图像上应用自动对比度。首先计算图像的直方图，将直方图中最亮像素的值映射为255，将直方图中最暗像素的值映射为0。

    参数：
        - **cutoff** (float, 可选) - 输入图像直方图中最亮和最暗像素的百分比。该值必须在 [0.0, 50.0) 范围内。默认值：0.0。
        - **ignore** (Union[int, sequence], 可选) - 要忽略的背景像素值，忽略值必须在 [0, 255] 范围内。默认值：None。

    异常：
        - **TypeError** - 如果 `cutoff` 不是float类型。
        - **TypeError** - 如果 `ignore` 不是int或sequence类型。
        - **ValueError** - 如果 `cutoff` 不在[0, 50.0) 范围内。
        - **ValueError** - 如果 `ignore` 不在[0, 255] 范围内。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
