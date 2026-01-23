mindspore.mint.addbmm
=====================

.. py:function:: mindspore.mint.addbmm(input, batch1, batch2, *, beta=1, alpha=1)

    对 `batch1` 和 `batch2` 应用批量矩阵乘法后进行归约加， `input` 和最终的结果相加。
    `alpha` 是 `batch1` 和 `batch2` 矩阵乘法的乘数， `beta` 是 `input` 的乘数。如果 `beta` 是0，那么 `input` 将会被忽略。

    .. math::
        output = \beta input + \alpha (\sum_{i=0}^{b-1} {batch1_i @ batch2_i})

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 被添加的Tensor。
        - **batch1** (Tensor) - 矩阵乘法中的第一个Tensor。
        - **batch2** (Tensor) - 矩阵乘法中的第二个Tensor。

    关键字参数：
        - **beta** (Union[int, float]，可选) - `input` 的乘数。默认值： ``1`` 。
        - **alpha** (Union[int, float]，可选) - `batch1` @ `batch2` 的乘数。默认值： ``1`` 。

    返回：
        Tensor，和 `input` 具有相同的dtype。

    异常：
        - **TypeError** - 如果 `alpha`， `beta` 不是int或者float。
        - **ValueError** - 如果 `batch1`， `batch2` 不能进行批量矩阵乘法。
        - **ValueError** - 如果 `batch1` 或 `batch2` 的不是三维Tensor。
