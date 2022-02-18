mindspore.dataset.vision.py_transforms.Cutout
=============================================

.. py:class:: mindspore.dataset.vision.py_transforms.Cutout(length, num_patches=1)

    随机去除输入numpy.ndarray图像上一定数量的正方形区域，将区域内像素值置为0。

    请参阅论文 `Improved Regularization of Convolutional Neural Networks with Cutout <https://arxiv.org/pdf/1708.04552.pdf>`_ 。

    **参数：**

    - **length** (int) - 去除正方形区域的边长。
    - **num_patches** (int，可选) - 去除区域的数量，默认值：1。
    
    **异常：**

    - **TypeError** - 当 `length` 的类型不为整型。
    - **TypeError** - 当 `num_patches` 的类型不为整型。
    - **ValueError** - 当 `length` 小于等于0。
    - **ValueError** - 当 `num_patches` 小于等于0。
    - **RuntimeError** - 当输入图像的shape不为<H, W, C>。
