mindspore.dataset.vision.Inter
==============================

.. py:class:: mindspore.dataset.vision.Inter

    图像插值方式枚举类。

    可选枚举值为：Inter.NEAREST、Inter.ANTIALIAS、Inter.LINEAR、Inter.BILINEAR、Inter.CUBIC、Inter.BICUBIC、Inter.AREA、Inter.PILCUBIC。

    - **Inter.Nest** - 最近邻插值。
    - **Inter.ANTIALIAS** - 抗锯齿插值。
    - **Inter.LINEAR** - 线性插值，实现同Inter.BILINEAR。
    - **Inter.BILINEAR** - 是双线性插值。
    - **Inter.CUBIC** - 三次插值，实现同Inter.BICUBIC。
    - **Inter.BICUBIC** - 双三次插值。
    - **Inter.AREA** - 像素区域插值。
    - **Inter.PILCUBIC** - Pillow库中实现的双三次插值，输入需为3通道格式。
