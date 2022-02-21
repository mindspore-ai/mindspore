mindspore.dataset.vision.c_transforms.RandomSharpness
=====================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomSharpness(degrees=(0.1, 1.9))

    在给定的范围随机内地调整输入图像的锐度。调节系数为0.0时将返回模糊图像；调节系数为1.0时将返回原始图像；调节系数为2.0时将返回锐化图像。

    **参数：**

    - **degrees** (Union[list, tuple], optional) - 锐度调节系数的随机选取范围，按照(min, max)顺序排列。如果 min=max，那么它是一个单一的固定锐度调整操作，默认值：(0.1, 1.9)。

    **异常：**

    - **TypeError** - 如果 `degree` 不是列表或元组。
    - **ValueError** - 如果 `degree` 为负数。
    - **ValueError** - 如果 `degree` 采用 (max, min) 格式而不是 (min, max)。
