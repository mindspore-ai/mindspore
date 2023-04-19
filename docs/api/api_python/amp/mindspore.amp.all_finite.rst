mindspore.amp.all_finite
========================

.. py:function:: mindspore.amp.all_finite(inputs)

    检查inputs是否是有效值（无溢出）。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

        此接口只在整网训练情况下用于判断梯度是否溢出，返回结果在不同后端可能存在差异。

    参数：
        - **inputs** (Union(tuple(Tensor), list(Tensor))) - 可迭代的Tensor。

    返回：
        Tensor，布尔类型的标量Tensor。
    
    教程样例：
        - `自动混合精度 - 损失缩放
          <https://mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html#损失缩放>`_
