mindspore.ops.cholesky_solve
============================

.. py:function:: mindspore.ops.cholesky_solve(input, input2, upper=False)

    根据Cholesky分解因子 `input2` 计算一组具有正定矩阵的线性方程组的解。

    如果 `upper` 为 ``True``， `input2` 是上三角矩阵，输出的结果：

    .. math::
        output = (input2^{T} * input2)^{{-1}}input

    如果 `upper` 为 ``False``， `input2` 是下三角矩阵，输出的结果：

    .. math::
        output = (input2 * input2^{T})^{{-1}}input

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - shape为 :math:`(*, N, M)` 的Tensor，表示2D或3D矩阵，数据类型是float32或float64。
        - **input2** (Tensor) - shape为 :math:`(*, N, N)` 的Tensor，表示由2D或3D方阵组成上三角或下三角的Cholesky因子，数据类型是float32或float64。 `input` 和 `input2` 必须具有相同的数据类型。
        - **upper** (bool, 可选) - 标志，将Cholesky因子视为上三角矩阵或下三角矩阵。默认值：``False``，Cholesky因子为下三角矩阵。

    返回：
        Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `upper` 不是bool。
        - **TypeError** - 如果 `input` 和 `input2` 的数据类型不是float32或float64。
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input2` 不是Tensor。
        - **ValueError** - 如果 `input` 和 `input2` 的批次大小相同。
        - **ValueError** - 如果 `input` 和 `input2` 的行数不同。
        - **ValueError** - 如果 `input` 不是2D或3D的矩阵。
        - **ValueError** - 如果 `input2` 不是2D或3D的方阵。
