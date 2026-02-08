mindspore.ops.coo_tan
======================

.. py:function:: mindspore.ops.coo_tan(x)

    计算COOTensor输入元素的正切值。

    .. math::
        out_i = \tan(x_i)

    参数：
        - **x** (COOTensor) - Tan的输入，任意维度的COOTensor。

    返回：
        COOTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
