mindspore.ops.addr
==================

.. py:function:: mindspore.ops.addr(x, vec1, vec2, *, beta=1, alpha=1)

    计算 `vec1` 和 `vec2` 的外积，并加到 `x` 中。

    .. note::
        - 如果 `vec1` 是一个大小为 :math:`N` 的向量， `vec2` 是一个大小为 :math:`M` 的向量，
          则 `x` 必须能够与大小为 :math:`(N, M)` 的矩阵进行广播，且输出将是大小为 :math:`(N, M)` 的矩阵。
        - 若 `beta` 为0，那么 `input` 将会被忽略。

    .. math::
        output = β x + α (vec1 ⊗ vec2)

    参数：
        - **x** (Tensor) - 输入tensor。
        - **vec1** (Tensor) - 将被乘的向量。
        - **vec2** (Tensor) - 将被乘的向量。

    关键字参数：
        - **beta** (scalar[int, float, bool], 可选) - `x` 的尺度因子。默认 ``1`` 。
        - **alpha** (scalar[int, float, bool], 可选) - （ `vec1` ⊗ `vec2` ）的尺度因子。默认 ``1`` 。

    返回：
        Tensor