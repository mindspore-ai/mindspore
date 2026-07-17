mindspore.ops.coo_atanh
========================

.. py:function:: mindspore.ops.coo_atanh(x)

    逐元素计算输入COOTensor的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(x_{i})

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (COOTensor) - 输入COOTensor。数据类型支持：float16、float32。

    返回：
        COOTensor，数据类型与输入相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
        - **TypeError** - 如果 `x` 的数据类型既不是float16也不是float32。
