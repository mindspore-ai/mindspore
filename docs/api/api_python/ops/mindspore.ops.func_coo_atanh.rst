mindspore.ops.coo_atanh
========================

.. py:function:: mindspore.ops.coo_atanh(x)

    逐元素计算输入COOTensor的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(x_{i})

    .. warning::
        这是一个实验性API，后续可能会修改或删除。

    参数：
        - **x** (COOTensor) - 输入COOTensor。数据类型支持：float16、float32。

    返回：
        COOTensor的数据类型与输入相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
        - **TypeError** - 如果 `x` 的数据类型既不是float16也不是float32。
