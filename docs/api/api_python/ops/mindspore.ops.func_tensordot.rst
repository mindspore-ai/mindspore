mindspore.ops.tensordot
=======================

.. py:function:: mindspore.ops.tensordot(a, b, dims=2)

    在指定轴 `dims` 上对Tensor `a` 和 `b` 进行点乘操作。

    参数：
        - **a** (Tensor) - 第一个输入Tensor。
        - **b** (Tensor) - 第二个输入Tensor。
        - **dims** (Union[int, tuple(tuple(int)), list(list(int))], 可选) - 指定 `a` 和 `b` 计算轴。`dims` 可为单个值，也可为长度为2的tuple或list。`dims` 是长度为2的tuple或list时，`dims` 的第一个成员是 `a` 选定的维度，第二个成员是 `b` 选定的维度。两个输入中选定的维度必须相互匹配。`a` 和 `b` 指定的维度个数必须相同，并且值在 `a` 和 `b` 的维度数的范围内。当 `dims` 是 ``N`` 的值时，指定 `a` 的后 ``N`` 个维度 和 `b` 的前 ``N`` 维进行点乘。默认值：``2``。

            - :math:`dims=0` 计算外积。
            - :math:`dims=1` 与 :math:`dims=((1,),(0,))` 相同，其中 `a` 和 `b` 都是2D。
            - :math:`dims=2` 与 :math:`dims=((1,2),(0,1))` 相同，其中 `a` 和 `b` 都是3D的。

    返回：
        Tensor，输出Tensor的shape为 :math:`(N + M)` 。其中 :math:`N` 和 :math:`M` 在两个输入中没有计算，是自由轴。

    异常：
        - **TypeError** - `a` 或 `b` 不是Tensor。
        - **TypeError** - `dims` 不是int、tuple或list。
