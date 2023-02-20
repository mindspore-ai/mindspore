mindspore.ops.addr
==================

.. py:function:: mindspore.ops.addr(x, vec1, vec2, beta=1, alpha=1)

    计算 `vec1` 和 `vec2` 的外积，并将其添加到 `x` 中。

    如果 `vec1` 是一个大小为 :math:`N` 的向量， `vec2` 是一个大小为 :math:`M` 的向量，那么 `x` 必须可以和大小为 :math:`(N, M)` 的矩阵广播，同时返回是一个大小为 :math:`(N, M)` 的矩阵。

    可选值 `bata` 和 `alpha` 分别是 `vec1` 和 `vec2` 外积的扩展因子以及附加矩阵 `x` 。如果 `beta` 为0，那么 `x` 将被忽略。

    .. math::
        output = β x + α (vec1 ⊗ vec2)

    参数：
        - **x** (Tensor) - 需要相加的向量。Tensor的shape是 :math:`(N, M)` 。
        - **vec1** (Tensor) - 第一个需要相乘的Tensor，shape大小为 :math:`(N,)` 。
        - **vec2** (Tensor) - 第二个需要相乘的Tensor，shape大小为 :math:`(M,)` 。
        - **beta** (scalar[int, float, bool], 可选) - `x` (β)的乘数。 `beta` 必须是int或float或bool类型，默认值：1。
        - **alpha** (scalar[int, float, bool], 可选) - `vec1` ⊗ `vec2` (α)的乘法器。 `alpha` 必须是int或float或bool类型，默认值：1。

    返回：
        Tensor，shape大小为 :math:`(N, M)` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 、 `vec1` 、 `vec2` 不是Tensor。
        - **TypeError** - `x` 、 `vec1` 、 `vec2` 的数据类型不一致。
        - **ValueError** - 如果 `x` 不是一个二维Tensor。
        - **ValueError** - 如果 `vec1` ， `vec2` 不是一个一维Tensor。
