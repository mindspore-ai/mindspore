mindspore.ops.atleast_3d
=========================

.. py:function:: mindspore.ops.atleast_3d(inputs)

    调整 `inputs` 中的Tensor维度，使输入中每个Tensor维度不低于3。

    Scalar、一维或二维Tensor被转换为三维Tensor，而高维输入则被保留。

    参数：
        - **inputs** (Union[Tensor, list[Tensor]]) - 一个或多个输入Tensor。例如，shape为 :math:`(N,)` 的Tensor变成shape为 :math:`(1, N, 1)` 的Tensor，shape为 :math:`(M, N)` 的Tensor变成shape为 :math:`(M, N, 1)` 的Tensor。

    返回：
        Tensor或Tensor列表。如果返回列表，则列表中的每一个元素 `a` 满足： `a`.ndim >= 3。
        例如，一个shape为 :math:`(N,)` 的Tensor，操作后shape变成 :math:`(1, N, 1)` ，shape为 :math:`(M, N)` 的2-D Tensor shape变成 :math:`(M, N, 1)` 。

    异常：
        - **TypeError** - `input` 不是一个Tensor或Tensor列表。
