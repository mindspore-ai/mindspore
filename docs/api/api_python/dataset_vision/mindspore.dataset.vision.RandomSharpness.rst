mindspore.dataset.vision.RandomSharpness
========================================

.. py:class:: mindspore.dataset.vision.RandomSharpness(degrees=(0.1, 1.9))

    在固定或随机的范围调整输入图像的锐度。度数为0.0时将返回模糊图像；度数为1.0时将返回原始图像；度数为2.0时将返回锐化图像。

    参数：
        - **degrees** (Union[list, tuple], 可选) - 锐度调节系数的随机选取范围，需为非负数，按照(min, max)顺序排列。如果min与max相等，将使用固定的调节系数进行处理。默认值：(0.1, 1.9)。

    异常：
        - **TypeError** - 如果 `degree` 的类型不为list或tuple。
        - **ValueError** - 如果 `degree` 为负数。
        - **ValueError** - 如果 `degree` 采用 (max, min) 格式而不是 (min, max)。
