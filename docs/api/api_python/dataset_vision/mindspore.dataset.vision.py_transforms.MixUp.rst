mindspore.dataset.vision.py_transforms.MixUp
============================================

.. py:class:: mindspore.dataset.vision.py_transforms.MixUp(batch_size, alpha, is_single=True)

    随机混合一批输入的numpy.ndarray图像及其标签。

    首先将每个图像乘以一个从Beta分布随机生成的权重lambda，然后加上另一个图像与1 - lambda之积，最后使用同样的lambda值将图像对应的标签进行混合，请确保标签预先进行了one-hot编码。

    **参数：**

    - **batch_size** (int) - 批处理大小，即图片的数量。
    - **alpha** (float) - Beta分布的α参数值，β参数也将使用该值。
    - **is_single** (bool，可选) - 若为True，将在批内随机混合图像[img0, ..., img(n-1), img(n)]与[img1, ..., img(n), img0]及对应标签；否则，将每批图像与前一批图像的处理结果混合。默认值：True。

    **异常：**
        
    - **TypeError** - 当 `batch_size` 的类型不为整型。
    - **TypeError** - 当 `alpha` 的类型不为浮点型。
    - **TypeError** - 当 `is_single` 的类型不为布尔型。
    - **ValueError** - 当 `batch_size` 不为正数。
    - **ValueError** - 当 `alpha` 不为正数。
