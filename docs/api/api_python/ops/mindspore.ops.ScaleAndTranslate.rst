mindspore.ops.ScaleAndTranslate
================================

.. py:class:: mindspore.ops.ScaleAndTranslate(kernel_type='lanczos3', antialias=True)

    根缩放并平移输入图像Tensor。

    .. note::
        - 输入图像必须是4D Tensor。
        - 输入 `size` 、 `scale` 和 `translation` 必须是包含两个元素的一维Tensor。

    参数：
        - **kernel_type** (str，可选) - 决定选择哪种图像滤波算法。可选列表：[ ``"lanczos1"`` , ``"lanczos3"`` , ``"lanczos5"`` , ``"gaussian"`` , ``"box"`` , ``"triangle"`` , ``"keyscubic"`` , ``"mitchellcubic"`` ]。默认值： ``"lanczos3"`` 。
        - **antialias** (bool，可选) - 决定是否使用抗锯齿。默认值： ``True`` 。

    输入：
        - **images** (Tensor) - 四维Tensor，shape为 :math:`(batch, image\_height, image\_width, channel)` 。
        - **size** (Tensor) - 缩放和平移操作后输出图像的大小。包含两个正数的的一维Tensor，形状必须为 :math:`(2,)` ，数据类型为int32。
        - **scale** (Tensor) - 指示缩放因子。包含两个正数的的一维Tensor，形状必须为 :math:`(2,)` ，数据类型为int32。
        - **translation** (Tensor) - 平移像素值。包含两个数的的一维Tensor，形状必须为 :math:`(2,)` ，数据类型为float32。

    输出：
        4-D Tensor，其shape为 :math:`(batch, size[0], size[1], channel)` ，数据类型为float32。

    异常：
        - **TypeError** - `kernel_type` 不是str类型。
        - **TypeError** - `antialias` bool类型。
        - **TypeError** - `images` 数据类型无效。
        - **TypeError** - `size` 不是int32类型。
        - **TypeError** - `scale` 不是float32类型。
        - **TypeError** - `translation` 不是Tensor或者数据类型不是float32。
        - **ValueError** - `kernel_type` 不在列表里面：["lanczos1", "lanczos3", "lanczos5", "gaussian", "box", "triangle", "keyscubic", "mitchellcubic"]。
        - **ValueError** - `images` 的秩不等于4。
        - **ValueError** - `size` 的shape不是 :math:`(2,)` 。
        - **ValueError** - `scale` 的shape不是 :math:`(2,)` 。 
        - **ValueError** - `translation` 的shape不是 :math:`(2,)` 。

