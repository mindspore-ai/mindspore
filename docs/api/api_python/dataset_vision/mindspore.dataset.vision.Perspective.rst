mindspore.dataset.vision.Perspective
====================================

.. py:class:: mindspore.dataset.vision.Perspective(start_points, end_points, interpolation=Inter.BILINEAR)

    对输入图像进行透视变换。

    参数：
        - **start_points** (Sequence[Sequence[int, int]]) - 起始点坐标序列，包含四个两元素子序列，分别对应原图中四边形的 [左上、右上、右下、左下]。
        - **end_points** (Sequence[Sequence[int, int]]) - 目标点坐标序列，包含四个两元素子序列，分别对应目标图中四边形的 [左上、右上、右下、左下]。
        - **interpolation** (:class:`~.vision.Inter`，可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.BILINEAR``。

    异常：
        - **TypeError** - 如果 `start_points` 不是Sequence[Sequence[int, int]]类型。
        - **TypeError** - 如果 `end_points` 不是Sequence[Sequence[int, int]]类型。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
