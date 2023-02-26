mindspore.ops.orgqr
====================

.. py:function:: mindspore.ops.orgqr(x, tau)

    计算 :class:`mindspore.ops.Geqrf` 返回的正交矩阵 :math:`Q` 的显式表示。

    下面以输入无batch维的情况为例， 计算 `Householder <https://en.wikipedia.org/wiki/Householder_transformation#Householder_matrix>`_ 矩阵的前 :math:`N` 列。
    假设输入 `x` 的shape经过 `Householder转换 <https://en.wikipedia.org/wiki/Householder_transformation#Householder_matrix>`_ 之后为：:math:`(M, N)` 。
    当 `x` 的对角线被置为1， `x` 中下三角形的每一列都表示为： :math:`w_j` ，其中 :math:`j` 在 :math:`j=1, \ldots, M` 范围内，此函数返回Householder矩阵乘积的前 :math:`N` 列：

    .. math::
        H_{1} H_{2} \ldots H_{k} \quad \text { with } \quad H_{j}=\mathrm{I}_{M}-\tau_{j} w_{j} w_{j}^{\mathrm{H}}

    其中：:math:`\mathrm{I}_{M}` 是 :math:`M` 维单位矩阵。当 :math:`w` 是复数的时候，:math:`w^{\mathrm{H}}` 是共轭转置，否则是一般转置。输出矩阵的shape与输入矩阵 `x` 相同。    

    参数：
        - **x** (Tensor) - shape :math:`(*, M, N)` 的Tensor，表示二维或者三维矩阵。数据类型为float32、float64、complex64或者complex128。
        - **tau** (Tensor) - Householder转换的反射系数，其shape为 :math:`(*, K)` ，其中 `K` 小于等于 `N` 。数据类型与 `x` 一致。

    返回：
        Tensor，数据类型与shape与 `x` 一致。

    异常：
        - **TypeError** - `x` 或者 `tau` 不是Tensor。
        - **TypeError** -  `x` 和 `tau` 的数据类型不是float64、float32、complex64或者complex128。
        - **ValueError** -  `x` 和 `tau` 的batch维度不同。
        - **ValueError** - `x`.shape[-2] < `x`.shape[-1]。
        - **ValueError** - `x`.shape[-1] < `tau`.shape[-1]。
        - **ValueError** - rank(`x`) - rank(`tau`) != 1。
        - **ValueError** - rank(`x`) != 2 or 3。
