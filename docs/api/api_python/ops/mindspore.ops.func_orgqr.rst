mindspore.ops.orgqr
====================

.. py:function:: mindspore.ops.orgqr(input, input2)

    计算Householder矩阵的前 :math:`N` 列。

    通常用于计算 :class:`mindspore.ops.Geqrf` 返回的正交矩阵 :math:`Q` 的显式表示。

    下面以输入无batch维的情况为例：
    假设输入 `input` 的shape经过Householder转换之后为：:math:`(M, N)` 。
    当 `input` 的对角线被置为1， `input` 中下三角形的每一列都表示为： :math:`w_j` ，其中 :math:`j` 在 :math:`j=1, \ldots, M` 范围内，此函数返回Householder矩阵乘积的前 :math:`N` 列：

    .. math::
        H_{1} H_{2} \ldots H_{k} \quad \text { with } \quad H_{j}=\mathrm{I}_{M}-\tau_{j} w_{j} w_{j}^{\mathrm{H}}

    其中：:math:`\mathrm{I}_{M}` 是 :math:`M` 维单位矩阵。当 :math:`w` 是复数的时候，:math:`w^{\mathrm{H}}` 是共轭转置，否则是一般转置。输出的shape与输入shape相同。
    :math:`\tau` 即输入 `input2` 。

    参数：
        - **input** (Tensor) - 二维或三维输入tensor，表示Householder反射向量，shape :math:`(*, M, N)` 。
        - **input2** (Tensor) - 一维或二维输入tensor，表示Householder反射系数，shape为 :math:`(*, K)` ，其中 `K` 小于等于 `N` 。
    返回：
        Tensor