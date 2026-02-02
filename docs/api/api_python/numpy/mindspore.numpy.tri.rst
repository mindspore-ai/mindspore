mindspore.numpy.tri
=================================

.. py:function:: mindspore.numpy.tri(N, M=None, k=0, dtype=mstype.float32)

    返回一个Tensor，在给定的对角线处及以下元素值为1，在其他位置为0。

    参数：
        - **N** (int) - 输入数组的行数。
        - **M** (int, 可选) - 输入数组的列数。默认情况下 `M` 等于 `N` 。
        - **k** (int, 可选) - 对角线的偏移量： :math:`k=0` 即为主对角线， :math:`k<0` 即对角线向下偏移， :math:`k>0` 即对角线向上偏移。默认值： ``0`` 。
        - **dtype** (mindspore.dtype, 可选) - 指定的Tensor数据类型。默认值： ``mstype.float32`` 。

    返回：
        Tensor，shape为 `(N, M)` ，其下三角区域填充为1，其余位置填充为0；用公式表达就是当 :math:`j<=i+k` 时， :math:`T[i,j]=1` ，否则为0。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
