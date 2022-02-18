mindspore.dataset.vision.py_transforms.AutoContrast
===================================================

.. py:class:: mindspore.dataset.vision.py_transforms.AutoContrast(cutoff=0.0, ignore=None)

    最大化（标准化）输入PIL图像的对比度。

    首先计算输入图像的直方图，移除指定 `cutoff` 比例的最亮和最暗像素后，将像素值重新映射至[0, 255]，使得最暗像素变为黑色，最亮像素变为白色。

    **参数：**

    - **cutoff** (float，可选) - 从直方图中移除最亮和最暗像素的百分比，取值范围为[0.0, 50.0)，默认值：0.0。
    - **ignore** (Union[int, sequence]，可选) - 背景像素值，将会被直接映射为白色，默认值：None，表示没有背景像素。

    **异常：**

    - **TypeError** - 当 `cutoff` 的类型不为浮点型。
    - **TypeError** - 当 `ignore` 的类型不为整型或序列。
    - **ValueError** - 当 `cutoff` 取值不在[0, 50.0)范围内。
    - **ValueError** - 当 `ignore` 取值不在[0, 255]范围内。
    - **RuntimeError** - 当输入图像的shape不为<H, W>或<H, W, C>。
