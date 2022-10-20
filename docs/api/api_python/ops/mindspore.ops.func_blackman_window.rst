mindspore.ops.blackman_window
=============================

.. py:function:: mindspore.ops.blackman_window(window_length, periodic=True, dtype=mstype.float32)

    布莱克曼窗口函数。

    `window_length` 是一个Tensor，控制返回的窗口大小，其数据类型必须是整数。特别的，当 `window_length` 为1时，返回的窗口只包含一个值，为 `1` 。`periodic` 决定返回的窗口是否会删除对称窗口的最后一个重复值，并准备用作带函数的周期窗口。因此，如果 `periodic` 为True，the :math:`N` 为 :math:`window\_length + 1`。

    .. math::
        w[n] = 0.42 - 0.5 cos(\frac{2\pi n}{N - 1}) + 0.08 cos(\frac{4\pi n}{N - 1})

        \text{where : N is the full window size.}

    参数：
        - **window_length** (Tensor) - 返回窗口的大小，数据类型为int32，int64。输入数据的值为[0,1000000]的整数。
        - **periodic** (bool) - 如果为True，返回一个窗口作为周期函数使用。如果为False，返回一个对称窗口。默认值：True。
        - **dtype** (mindspore.dtype) - 输出数据类型，目前只支持float16、float32和float64。默认值：mindspore.float32。

    返回：
        1D Tensor，大小为 `window_length` ，数据类型与 `dtype` 一致。

    异常：
        - **TypeError** - `window_length` 不是Tensor。
        - **TypeError** - `periodic` 不是bool。
        - **TypeError** - `dtype` 不是float16、float32、float64。
        - **TypeError** - `window_length` 的数据类型不是int32、int64。
        - **ValueError** - `window_length` 的值不在[0, 1000000]。
        - **ValueError** - `window_length` 的维度不等于0。
