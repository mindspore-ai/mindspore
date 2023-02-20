mindspore.ops.addbmm
=====================

.. py:function:: mindspore.ops.addbmm(x, batch1, batch2, *, beta=1, alpha=1)

    对 `batch1` 和 `batch2` 应用批量矩阵乘法后进行reduced add， `x` 和最终的结果相加。
    `alpha` 和 `beta` 分别是 `batch1` 和 `batch2` 矩阵乘法和 `x` 的乘数。如果 `beta` 是0，那么 `x` 将会被忽略。

    .. math::
        output = \beta x + \alpha (\sum_{i=0}^{b-1} {batch1 @ batch2})

    参数：
        - **x** (Tensor) - 被添加的tensor。
        - **batch1** (Tensor) - 矩阵乘法中的第一个张量。
        - **batch2** (Tensor) - 矩阵乘法中的第二个张量。

    关键字参数：
        - **beta** (Union[int, float]，可选) - `x` 的乘数。默认值：1。
        - **alpha** (Union[int, float]，可选) - `batch1` @ `batch2` 的乘数。默认值：1。

    返回：
        Tensor，和 `x` 具有相同的dtype。

    异常：
        - **TypeError** - 如果 `alpha`，`beta` 不是int或者float。
        - **ValueError** - 如果 `batch1`， `batch2` 不能进行批量矩阵乘法。
