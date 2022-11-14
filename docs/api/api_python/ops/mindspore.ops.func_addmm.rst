mindspore.ops.addmm
====================

.. py:function:: mindspore.ops.addmm(x, mat1, mat2, *, beta=1, alpha=1)

    对 `mat1` 和 `mat2` 应用矩阵乘法。矩阵 `x` 和最终的结果相加。 `alpha` 和 `beta` 分别是 `mat1` 和 `mat2` 矩阵乘法和 `x` 的乘数。如果 `beta` 是0，那么 `x` 将会被忽略。

    .. math::
        output = \beta x + \alpha (mat1 @ mat2)

    参数：
        - **x** (Tensor) - 被添加的tensor。
        - **mat1** (Tensor) - 矩阵乘法中的第一个张量。
        - **mat2** (Tensor) - 矩阵乘法中的第二个张量。

    关键字参数：
        - **beta** (Union[int, float]，可选) - `x` 的乘数。默认值：1。
        - **alpha** (Union[int, float]，可选) - `mat1` @ `mat2` 的乘数。默认值：1。

    返回：
        Tensor，和 `x` 具有相同的dtype。

    异常：
        - **ValueError**：If `mat1`，`mat2` 不能进行矩阵乘法。