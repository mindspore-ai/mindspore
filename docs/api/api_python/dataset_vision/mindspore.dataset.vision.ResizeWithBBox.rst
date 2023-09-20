mindspore.dataset.vision.ResizeWithBBox
=======================================

.. py:class:: mindspore.dataset.vision.ResizeWithBBox(size, interpolation=Inter.LINEAR)

    将输入图像调整为给定的尺寸大小并相应地调整边界框的大小。

    参数：
        - **size** (Union[int, Sequence[int]]) - 图像的输出尺寸大小。若输入整型，将调整图像的较短边长度为 `size` ，且保持图像的宽高比不变；若输入是2元素组成的序列，其输入格式需要是 (高度, 宽度) 。
        - **interpolation** (:class:`~.vision.Inter`, 可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.LINEAR``。

    异常：
        - **TypeError** - 当 `size` 的类型不为int或Sequence[int]。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **ValueError** - 当 `size` 不为正数。
        - **RuntimeError** - 如果输入的Tensor不是 <H, W> 或 <H, W, C> 格式。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
