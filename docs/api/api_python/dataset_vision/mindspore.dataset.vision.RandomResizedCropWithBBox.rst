mindspore.dataset.vision.RandomResizedCropWithBBox
==================================================

.. py:class:: mindspore.dataset.vision.RandomResizedCropWithBBox(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)

    对输入图像进行随机裁剪且随机调整纵横比，并将处理后的图像调整为指定的尺寸大小，并相应地调整边界框。

    参数：
        - **size** (Union[int, Sequence[int]]) - 图像的输出尺寸大小。若输入整型，则放缩至(size, size)大小；若输入2元素序列，则以2个元素分别为高和宽放缩至(高度, 宽度)大小。
        - **scale** (Union[list, tuple], 可选) - 裁剪子图的尺寸大小相对原图比例的随机选取范围，需要在[min, max)区间。默认值：(0.08, 1.0)。
        - **ratio** (Union[list, tuple], 可选) - 裁剪子图的宽高比的随机选取范围，需要在[min, max)区间。默认值：(3./4., 4./3.)。
        - **interpolation** (:class:`mindspore.dataset.vision.Inter` , 可选) - 插值方式。它可以是 [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC] 中的任何一个。默认值：Inter.BILINEAR。

          - Inter.BILINEAR，双线性插值。
          - Inter.NEAREST，最近邻插值。
          - Inter.BICUBIC，双三次插值。

        - **max_attempts** (int, 可选) - 生成随机裁剪位置的最大尝试次数，超过该次数时将使用中心裁剪。默认值：10。

    异常：
        - **TypeError** - 当 `size` 的类型不为int或Sequence[int]。
        - **TypeError** - 当 `scale` 的类型不为tuple或list。
        - **TypeError** - 当 `ratio` 的类型不为tuple或list。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **TypeError** - 当 `max_attempts` 的类型不为int。
        - **ValueError** - 当 `size` 不为正数。
        - **ValueError** - 当 `scale` 为负数。
        - **ValueError** - 当 `ratio` 为负数。
        - **ValueError** - 当 `max_attempts` 不为正数。
        - **RuntimeError** 当输入图像的shape不为<H, W>或<H, W, C>。
