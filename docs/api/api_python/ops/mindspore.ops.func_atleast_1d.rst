mindspore.ops.atleast_1d
=========================

.. py:function:: mindspore.ops.atleast_1d(inputs)

    调整 `inputs` 中的Tensor维度，使输入中每个Tensor维度不低于1。

    Scalar输入被转换为一维Tensor，而高维输入则被保留。

    参数：
        - **inputs** (Union[Tensor, list[Tensor]]) - 一个或多个输入Tensor。

    返回：
        Tensor或Tensor列表。如果返回列表，则列表中的每一个元素 `a` 满足： `a`.ndim >= 1。

    异常：
        - **TypeError** - `input` 不是一个Tensor或Tensor列表。
