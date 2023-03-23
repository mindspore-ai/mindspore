mindspore.ops.coo_atanh
========================

.. py:function:: mindspore.ops.coo_atanh(x: COOTensor)

    逐元素计算输入COOTensor的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(x_{i})

    .. warning::
       实验性算子，将来有可能更改或删除。

    参数：
        - **x** (COOTensor) - COOTensor的输入。数据类型支持：float16、float32。

    返回：
        COOTensor的数据类型与输入相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16或float32。
