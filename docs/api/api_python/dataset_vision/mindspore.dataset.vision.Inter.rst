mindspore.dataset.vision.Inter
==============================

.. py:class:: mindspore.dataset.vision.Inter

    图像插值方法枚举类。

    可选值如下：

    - **Inter.NEAREST** - 最近邻插值。
    - **Inter.ANTIALIAS** - 抗锯齿插值。仅当输入为 `PIL.Image.Image` 时支持。
    - **Inter.LINEAR** - 线性插值，实现同 ``Inter.BILINEAR`` 。
    - **Inter.BILINEAR** - 双线性插值。
    - **Inter.CUBIC** - 三次插值，实现同 ``Inter.BICUBIC`` 。
    - **Inter.BICUBIC** - 双三次插值。
    - **Inter.AREA** - 像素区域插值。仅当输入为 `numpy.ndarray` 时支持。
    - **Inter.PILCUBIC** - 类Pillow实现的双三次插值。仅当输入为 `numpy.ndarray` 时支持。
