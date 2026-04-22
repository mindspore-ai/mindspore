mindspore.ops.ormqr
===================

.. py:function:: mindspore.ops.ormqr(input, tau, other, left=True, transpose=False)

    计算一个普通矩阵 `other` 与一个矩阵Q（由Householder反射向量 `input` 和反射系数 `tau` 生成）的乘积。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `tau` 将重命名为 `input2`，`other` 将重命名为 `input3`，并新增 `out` 参数。

    如果 `left` 为 ``True`` ，计算顺序为Q \* `other` ，否则，计算顺序为 `other` \* Q。

    参数：
        - **input** (Tensor) - 输入tensor，shape :math:`(*, mn, k)`，当 `left` 为 ``True`` 时， mn的值等于m，否则mn的值等于n。
        - **tau** (Tensor) - 输入tensor，shape :math:`(*, min(mn, k))` 。
        - **other** (Tensor) - 输入tensor，shape :math:`(*, m, n)` 。
        - **left** (bool, 可选) - 计算顺序。默认 ``True`` 。
        - **transpose** (bool, 可选) - 是否对矩阵Q进行共轭转置变换。默认 ``False`` 。

    返回：
        Tensor
