mindspore.numpy.log
===================

.. py:function:: mindspore.numpy.log(x, dtype=None)

    返回自然对数，逐元素计算。

    自然对数log是指数函数的逆函数，因此 ``log(exp(x)) = x`` 。自然对数是以e为底的对数。

    .. note::
        - 不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。
        - 在GPU上，支持的数据类型有np.float16和np.float32。
        - 在CPU上，支持的数据类型有np.float16，np.float32和np.float64。

    参数：
        - **x** (Tensor) - 输入数组。
        - **dtype** (mindspore.dtype, 可选) - 默认值: ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， `x` 的自然对数，逐元素计算。 如果 `x` 是标量，返回标量。
