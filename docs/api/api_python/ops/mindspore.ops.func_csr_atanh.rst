mindspore.ops.csr_atanh
========================

.. py:function:: mindspore.ops.csr_atanh(x)

    逐元素计算输入CSRTensor的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(x_{i})

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x** (CSRTensor) - CSRTensor的shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。数据类型支持：float16、float32。

    返回：
        CSRTensor，数据类型与输入相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
        - **TypeError** - `x` 的数据类型不是float16或float32。
