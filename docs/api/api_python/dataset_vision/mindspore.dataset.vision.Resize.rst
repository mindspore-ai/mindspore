mindspore.dataset.vision.Resize
============================================

.. py:class:: mindspore.dataset.vision.Resize(size, interpolation=Inter.LINEAR)

    对输入图像使用给定的 :class:`mindspore.dataset.vision.Inter` 插值方式去调整为给定的尺寸大小。

    参数：
        - **size** (Union[int, Sequence[int]]) - 图像的输出尺寸大小。若输入整型，将调整图像的较短边长度为 `size` ，且保持图像的宽高比不变；若输入是2元素组成的序列，其输入格式需要是 (高度, 宽度) 。
        - **interpolation** (:class:`~.vision.Inter`, 可选) - 图像插值方式。取值可以为 ``Inter.LINEAR`` 、 ``Inter.NEAREST`` 、 ``Inter.BICUBIC`` 或 ``Inter.PILCUBIC`` 。默认值： ``Inter.LINEAR`` 。

          - ``Inter.BILINEAR``，双线性插值。
          - ``Inter.LINEAR``，双线性插值，同 Inter.BILINEAR 。
          - ``Inter.NEAREST``，最近邻插值。
          - ``Inter.BICUBIC``，双三次插值。
          - ``Inter.AREA``，像素区域插值。
          - ``Inter.PILCUBIC``，双三次插值，实现同Pillow，仅当输入为numpy.ndarray格式的3通道图像时有效。
          - ``Inter.ANTIALIAS``，抗锯齿插值。

    异常：
        - **TypeError** - 当 `size` 的类型不为int或Sequence[int]。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **ValueError** - 当 `size` 不为正数。
        - **RuntimeError** - 如果输入的Tensor不是 <H, W> 或 <H, W, C> 格式。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
