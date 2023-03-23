mindspore.ops.ConjugateTranspose
=================================

.. py:class:: mindspore.ops.ConjugateTranspose

    计算 `x` 按 `perm` 进行行列变换后的共轭矩阵。

    .. math::
        y[i, j, k, ..., s, t, u] == Conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]]), i, j, ... ∈ [0, rank(x))

    输入：
        - **x** (Tensor) - 输入要计算的Tensor。
        - **perm** (tuple[int]) - 进行行列变换时输入与输出的索引对应规则。 `perm` 由 `x` 每个维度的索引组成，其长度必须和 `x` 的shape长度相同，且仅支持常量值。

    输出：
        Tensor。输出Tensor的shape由输入的Tensor与 `perm` 决定：

        .. math::
            y.shape[i] = x.shape[perm[i]]
        
        其中i在[0, rank(x) - 1]范围内。

    异常：
        - **TypeError** - `perm` 不是tuple格式。
        - **ValueError** - `x` 和 `perm` 的shape长度不一致。
        - **ValueError** - `perm` 中存在相同元素。
