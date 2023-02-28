mindspore.amp.all_finite
========================

.. py:function:: mindspore.amp.all_finite(inputs, status=None)

    检查inputs是否是有效值（无溢出）。

    .. note::
        这是一个实验性接口，后续可能删除或修改。

        此接口只在整网训练情况下用于判断梯度是否溢出，返回结果在不同后端可能存在差异。

    参数：
        - **inputs** (Union(tuple(Tensor), list(Tensor))) - 可迭代的Tensor。

    返回：
        Tensor，布尔类型的标量Tensor。
