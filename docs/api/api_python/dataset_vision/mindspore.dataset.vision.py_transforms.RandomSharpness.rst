mindspore.dataset.vision.py_transforms.RandomSharpness
======================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomSharpness(degrees=(0.1, 1.9))

    随机调整输入PIL图像的锐度。

    **参数：**

    - **degrees** (sequence) - 锐度调节系数的随机选取范围，按照(min, max)顺序排列。调节系数为0.0时将返回模糊图像；调节系数为1.0时将返回原始图像；调节系数为2.0时将返回锐化图像。默认值：(0.1,1.9)。

    **异常：**

    - **TypeError** - 当 `degrees` 的类型不为序列。
    - **ValueError** - 当 `degrees` 为负数。
    - **ValueError** - 当 `degrees` 未按照(min, max)顺序排列。
