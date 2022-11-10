mindspore.ops.coo_isinf
========================

.. py:function:: mindspore.ops.coo_isinf(x: COOTensor)

    判断输入数据每个位置上的值是否是inf。

    参数：
        - **x** (COOTensor) - isinf的输入，COOTensor。

    返回：
        COOTensor，输出的shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 不是COOTensor。
