mindspore.ops.hamming_window
============================

.. py:function:: mindspore.ops.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None)

    返回一个Hamming window。

    .. math::
        w[n]=\alpha − \beta \cos \left( \frac{2 \pi n}{N - 1} \right),

    这里 :math:`N` 是整个window的大小。

    参数：
        - **window_length** (int) - 输出window的大小。为非负整数。
        - **periodic** (bool, 可选) - 如果为 ``True`` ，则返回周期性window。如果为 ``False`` ，则返回对称的window。默认值： ``True`` 。
        - **alpha** (float, 可选) - 系数α。默认值： ``0.54`` 。
        - **beta** (float, 可选) - 系数β。默认值： ``0.46`` 。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 输出window的数据类型。默认值： ``None`` 。
    
    返回：
        Tensor，包含输出window的大小为 `window_length` 的1-D Tensor。

    异常：
        - **TypeError** - 如果 `periodic` 不是bool。
        - **TypeError** - 如果 `window_length` 是负整数。
