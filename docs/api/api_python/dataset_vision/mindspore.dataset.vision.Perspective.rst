mindspore.dataset.vision.Perspective
====================================

.. py:class:: mindspore.dataset.vision.Perspective(start_points, end_points, interpolation=Inter.BILINEAR)

    对输入图像进行透视变换。

    参数：
        - **start_points** (Sequence[Sequence[int, int]]) - 起始点坐标序列，包含四个两元素子序列，分别对应原图中四边形的 [左上、右上、右下、左下]。
        - **end_points** (Sequence[Sequence[int, int]]) - 目标点坐标序列，包含四个两元素子序列，分别对应目标图中四边形的 [左上、右上、右下、左下]。
        - **interpolation** (Inter，可选) - 插值方式，取值可为 Inter.BILINEAR、Inter.LINEAR、Inter.NEAREST、Inter.AREA、Inter.PILCUBIC、Inter.CUBIC 或 Inter.BICUBIC。默认值：Inter.BILINEAR。

          - **Inter.BILINEAR**：双线性插值。
          - **Inter.LINEAR**：线性插值，与双线性插值相同。
          - **Inter.NEAREST**：最近邻插值。
          - **Inter.BICUBIC**：双三次插值。
          - **Inter.CUBIC**：三次插值，与双三次插值相同。
          - **Inter.PILCUBIC**：类Pillow实现的三次插值，只支持 :class:`numpy.ndarray` 类型输入。
          - **Inter.AREA**：像素区域插值，只支持 :class:`numpy.ndarray` 类型输入。

    异常：
        - **TypeError** - 如果 `start_points` 不是Sequence[Sequence[int, int]]类型。
        - **TypeError** - 如果 `end_points` 不是Sequence[Sequence[int, int]]类型。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W> 或 <H, W, C>。
