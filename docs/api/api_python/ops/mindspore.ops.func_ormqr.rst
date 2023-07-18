mindspore.ops.ormqr
===================

.. py:function:: mindspore.ops.ormqr(input, tau, other, left=True, transpose=False)

    计算一个普通矩阵与Householder矩阵的乘积。计算维度为(m, n)的矩阵C（由 `other` 给出）和一个矩阵Q的乘积，
    其中Q由Householder反射系数（`input`, `tau`）生成。返回一个Tensor。

    参数：
        - **input** (Tensor) - shape :math:`(*, mn, k)`，当 `left` 为 ``True`` 时， mn的值等于m，否则mn的值等于n。 `*` 表示Tensor在轴0上的长度为0或者大于0。
        - **tau** (Tensor) - shape :math:`(*, min(mn, k))`，其中 `*` 表示Tensor在轴0上的长度为0或者大于0，其类型与 `input` 相同。
        - **other** (Tensor) - shape :math:`(*, m, n)`，其中 `*` 表示Tensor在轴0上的长度为0或者大于0，其类型与 `input` 相同。
        - **left** (bool, 可选) - 决定了矩阵乘积运算的顺序。如果 `left` 为 ``True`` ，计算顺序为op(Q) \* `other` ，否则，计算顺序为 `other` \* op(Q)。默认值： ``True`` 。
        - **transpose** (bool, 可选) - 如果为 ``True`` ，对矩阵Q进行共轭转置变换，否则，不对矩阵Q进行共轭转置变换。默认值： ``False`` 。

    返回：
        Tensor，数据类型与shape与 `other` 一致。

    异常：
        - **TypeError** - 如果 `input` ，`tau` 或者 `other` 不是Tensor。
        - **TypeError** - 如果 `input` ， `tau` 和 `other` 的dtype不是float64、float32、complex64或者complex128。
        - **ValueError** - 如果 `input` 或 `other` 的维度小于2D。
        - **ValueError** - rank(`input`) - rank(`tau`) != 1。
        - **ValueError** - tau.shape[:-2] != input.shape[:-2]。
        - **ValueError** - other.shape[:-2] != input.shape[:-2]。
        - **ValueError** - 当 `left` 为 ``True`` 时，other.shape[-2] < tau.shape[-1]。
        - **ValueError** - 当 `left` 为 ``True`` 时，other.shape[-2] != input.shape[-2]。
        - **ValueError** - 当 `left` 为 ``False`` 时，other.shape[-1] < tau.shape[-1]。
        - **ValueError** - 当 `left` 为 ``False`` 时，other.shape[-1] != input.shape[-2]。
