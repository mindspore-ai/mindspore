mindspore.ops.bartlett_window
=============================

.. py:function:: mindspore.ops.bartlett_window(window_length, periodic=True, *, dtype=None)

    巴特利特窗口函数。

    `window_length` 是一个Tensor，控制返回的窗口大小，其数据类型必须是整数。特别的，当 `window_length` 为1时，返回的窗口只包含一个值，为 `1` 。 `periodic` 决定返回的窗口是否会删除对称窗口的最后一个重复值，并准备用作带函数的周期窗口。因此，如果 `periodic` 为True， :math:`N` 为 :math:`window\_length + 1`。

    .. math::

        w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
        \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
        2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
        \end{cases},

    其中，N是总的窗口长度 `window_length` 。

    参数：
        - **window_length** (Tensor) - 返回窗口的大小，数据类型为int32，int64。输入数据的值为[0, 1000000]的整数。
        - **periodic** (bool，可选) - 决定返回的窗口作为周期函数还是对称窗口。默认值：True。

    关键字参数：
        - **dtype** (mindspore.dtype，可选) - 输出数据类型，目前只支持float16、float32和float64。默认值：None。

    返回：
        1D Tensor，大小为 `window_length` ，数据类型与 `dtype` 一致。如果 `dtype` 为None，则数据类型为float32。

    异常：
        - **TypeError** - `window_length` 不是Tensor。
        - **TypeError** - `window_length` 的数据类型不是int32、int64。
        - **TypeError** - `periodic` 不是bool。
        - **TypeError** - `dtype` 不是float16、float32、float64。
        - **ValueError** - `window_length` 的值不在[0, 1000000]。
        - **ValueError** - `window_length` 的维度不等于0。
