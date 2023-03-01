mindspore.ops.logical_not
==========================

.. py:function:: mindspore.ops.logical_not(x)

    逐元素计算一个Tensor的逻辑非运算。

    .. math::
        out_{i} = \neg x_{i}

    参数：
        - **x** (Tensor) - 输入Tensor。 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape与 `x` 相同，数据类型为bool。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
