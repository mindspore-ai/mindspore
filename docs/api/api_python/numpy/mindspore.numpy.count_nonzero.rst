mindspore.numpy.count_nonzero
=============================

.. py:function:: mindspore.numpy.count_nonzero(x, axis=None, keepdims=False)

    计算Tensor `x` 中的非零值数量。

    参数：
        - **x** (Tensor) - 需要统计非零值数量的Tensor。
        - **axis** (Union[int,tuple], 可选) - 指定沿着哪些轴统计非零元素个数，可以是单个轴，也可以用tuple表示多个轴。此时沿完全展平的 `x` 计算非零值数量。默认值： ``None`` 。
        - **keepdims** (bool, 可选) - 如果设置为 ``True`` ，结果会保留计数所沿的 `axis` ，该维度的大小为1。若使用此选项，结果会广播到 和 `x` 同一个维度数。默认值： ``False`` 。

    返回：
        Tensor， 表示 `x` 沿给定 `axis` 的非零值数量。否则，返回 `x` 中非零值的总数。

    异常：
        - **TypeError** - 如果 `axis` 不是int或tuple。
        - **ValueError** - 如果 `axis` 不在范围[-x.ndim, x.ndim)内。
