mindspore.ops.bartlett_window
=============================

.. py:function:: mindspore.ops.bartlett_window(window_length, periodic=True, *, dtype=None)

    巴特利特窗口函数。

    该函数是一种三角形状的加权函数，通常用于平滑处理或频域分析信号。

    .. math::

        w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
        \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
        2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
        \end{cases},

    其中， :math:`N` 是总的窗口长度 `window_length` ，n为小于 :math:`N` 的自然数 [0, 1, ..., N-1]。

    参数：
        - **window_length** (Tensor) - 窗口的大小。
        - **periodic** (bool，可选) - 如果为 ``True`` ，表示返回周期窗口。如果为 ``False`` ，表示返回对称窗口。默认 ``True`` 。

    关键字参数：
        - **dtype** (mindspore.dtype，可选) - 指定数据类型。默认 ``None`` 。

    返回：
        一维tensor。