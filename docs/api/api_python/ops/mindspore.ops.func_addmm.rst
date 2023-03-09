mindspore.ops.addmm
====================

.. py:function:: mindspore.ops.addmm(input, mat1, mat2, *, beta=1, alpha=1)

    对 `mat1` 和 `mat2` 应用矩阵乘法。矩阵 `input` 和最终的结果相加。 `alpha` 和 `beta` 分别是 `mat1` 和 `mat2` 矩阵乘法和 `input` 的乘数。如果 `beta` 是0，那么 `input` 将会被忽略。

    .. math::
        output = \beta input + \alpha (mat1 @ mat2)

    参数：
        - **input** (Tensor) - 被添加的Tensor。
        - **mat1** (Tensor) - 矩阵乘法中的第一个Tensor。
        - **mat2** (Tensor) - 矩阵乘法中的第二个Tensor。

    关键字参数：
        - **beta** (Union[int, float]，可选) - `input` 的乘数。默认值：1。
        - **alpha** (Union[int, float]，可选) - `mat1` @ `mat2` 的乘数。默认值：1。

    返回：
        Tensor，和 `input` 具有相同的dtype。

    异常：
        - **ValueError**：如果 `mat1`，`mat2` 不能进行矩阵乘法。