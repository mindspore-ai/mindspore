mindspore.ops.logical_not
==========================

.. py:function:: mindspore.ops.logical_not(input)

    逐元素计算一个Tensor的逻辑非运算。

    .. math::
        out_{i} = \neg input_{i}

    参数：
        - **input** (Tensor) - 输入Tensor。 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape与 `input` 相同，数据类型为bool。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
