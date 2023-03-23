mindspore.ops.BartlettWindow
==============================

.. py:class:: mindspore.ops.BartlettWindow(periodic=True, dtype=mstype.float32)

    巴特利特窗口函数。

    更多参考详见 :func:`mindspore.ops.bartlett_window`。

    参数：
        - **periodic** (bool，可选) - 如果为True，返回一个窗口作为周期函数使用。如果为False，返回一个对称窗口。默认值：True。
        - **dtype** (mindspore.dtype，可选) - 输出数据类型，目前只支持float16、float32和float64。默认值：mstype.float32。

    输入：
        - **window_length** (Tensor) - 返回窗口的大小，数据类型为int32，int64。输入数据的值为[0, 1000000]的整数。

    输出：
        1D Tensor，大小为 `window_length` ，数据类型由 `dtype` 指定。
