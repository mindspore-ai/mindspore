mindspore.ops.blackman_window
=============================

.. py:function:: mindspore.ops.blackman_window(window_length, periodic=True, *, dtype=None)

    布莱克曼窗口函数。

    常用来为FFT截取有限长的信号片段。

    .. math::
        w[n] = 0.42 - 0.5 \cos(\frac{2\pi n}{N - 1}) + 0.08 \cos(\frac{4\pi n}{N - 1})

    其中， :math:`N` 是总的窗口长度 `window_length` ，n为小于 :math:`N` 的自然数 [0, 1, ..., N-1]。

    参数：
        - **window_length** (Tensor) - 窗口的大小。
        - **periodic** (bool，可选) - 如果为 ``True`` ，表示返回周期窗口。如果为 ``False`` ，表示返回对称窗口。默认 ``True`` 。

    关键字参数：
        - **dtype** (mindspore.dtype，可选) - 指定数据类型。默认 ``None`` 。

    返回：
        一维tensor。
