mindspore.ops.addmv
======================

.. py:function:: mindspore.ops.addmv(x, mat, vec, beta=1, alpha=1)

    `mat` 和 `vec` 相乘，且将输入向量 `x` 加到最终结果中。

    如果 `mat` 是一个大小为 :math:`(N, M)` Tensor， `vec` 一个大小为 :math:`M` 的一维Tensor，那么 `x` 必须是可广播的，且
    带有一个大小为 :math:`N` 的一维Tensor，同时 `out` 是一个大小为 :math:`N` 的一维Tensor。

    可选值 `bata` 和 `alpha` 分别是 `mat` 和 `vec` 矩阵向量的乘积和附加Tensor `x` 的扩展因子。如果 `beta` 为0，那么 `x` 将被忽略。

    .. math::
        output = β x + α (mat @ vec)

    参数：
        - **x** (Tensor) - 被加的向量，Tensor的shape大小为 :math:`(N,)`。
        - **mat** (Tensor) - 第一个需要相乘的Tensor，shape大小为 :math:`(N, M)` 。
        - **vec** (Tensor) - 第二个需要相乘的Tensor，shape大小为 :math:`(M,)` 。
        - **beta** (scalar[int, float, bool], 可选) - `x` (β)的乘法器。 `beta` 必须是int或float或bool类型，默认值：1。
        - **alpha** (scalar[int, float, bool], 可选) - `mat` @ `vec` (α)的乘法器。 `alpha` 必须是int或float或bool类型，默认值：1。

    返回：
        Tensor，shape大小为 :math:`(N,)` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `mat` 、 `vec` 、 `x` 不是Tensor。
        - **TypeError** - `mat` 、 `vec` 、 `x` 的数据类型不一致。
        - **ValueError** - 如果 `mat` 不是一个二维Tensor。如果 `vec` 不是一个一维Tensor。
