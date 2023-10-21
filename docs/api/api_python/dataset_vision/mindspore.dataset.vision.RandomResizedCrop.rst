mindspore.dataset.vision.RandomResizedCrop
==========================================

.. py:class:: mindspore.dataset.vision.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)

    对输入图像进行随机裁剪，并使用指定的 :class:`mindspore.dataset.vision.Inter` 插值方式去调整为指定的尺寸大小。

    .. note:: 如果输入图像不止一张，需要保证输入的多张图像尺寸大小一致。

    参数：
        - **size** (Union[int, Sequence[int]]) - 图像的输出尺寸大小。若输入整型，则放缩至(size, size)大小；若输入2元素序列，则以2个元素分别为高和宽放缩至(高度, 宽度)大小。
        - **scale** (Union[list, tuple], 可选) - 裁剪子图的尺寸大小相对原图比例的随机选取范围，需要在[min, max)区间。默认值： ``(0.08, 1.0)`` 。
        - **ratio** (Union[list, tuple], 可选) - 裁剪子图的宽高比的随机选取范围，需要在[min, max)区间。默认值： ``(3. / 4., 4. / 3.)`` 。
        - **interpolation** (:class:`~.vision.Inter`, 可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.BILINEAR``。
        - **max_attempts** (int, 可选) - 生成随机裁剪位置的最大尝试次数，超过该次数时将使用中心裁剪。默认值： ``10`` 。

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

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
