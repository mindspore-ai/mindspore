mindspore.dataset.vision.py_transforms.RandomColor
==================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomColor(degrees=(0.1, 1.9))

    随机调整输入PIL图像的色彩平衡。

    **参数：**

    - **degrees** (sequence) - 色彩调节系数的随机选取范围，需为一个2元素序列，按照(min, max)的顺序排列，默认值：(0.1, 1.9)。调节系数为1.0时返回原始图像；调节系数为0.0时返回黑白图像；取值越大，图像的亮度、对比度等越大。

    **异常：**
        
    - **TypeError** - 当 `degrees` 的类型不为浮点型序列。
    - **ValueError** - 当 `degrees` 为负数。
    - **RuntimeError** - 当输入图像的shape不为<H, W, C>。
