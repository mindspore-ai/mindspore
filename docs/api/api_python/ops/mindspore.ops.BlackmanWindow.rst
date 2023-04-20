mindspore.ops.BlackmanWindow
==============================

.. py:class:: mindspore.ops.BlackmanWindow(periodic=True, dtype=mstype.float32)

    布莱克曼窗口函数。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.blackman_window`。

    参数：
        - **periodic** (bool，可选) - 如果为 ``True`` ，返回一个窗口作为周期函数使用。如果为 ``False`` ，返回一个对称窗口。默认值： ``True`` 。
        - **dtype** (mindspore.dtype，可选) - 输出数据类型，目前只支持float16、float32和float64。默认值： ``mstype.float32`` 。

    输入：
        - **window_length** (Tensor) - 返回窗口的大小，数据类型为int32，int64。输入数据的值为[0, 1000000]的整数。

    输出：
        1D Tensor，大小为 `window_length` ，数据类型与 `dtype` 一致。
