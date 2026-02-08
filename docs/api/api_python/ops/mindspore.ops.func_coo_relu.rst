mindspore.ops.coo_relu
=======================

.. py:function:: mindspore.ops.coo_relu(x)

    对输入的COOTensor逐元素计算其应用ReLU激活函数后的值。

    返回 :math:`\max(x,\  0)` 的值。负值神经元将被设置为0，正值神经元将保持不变。

    .. math::
        ReLU(x) = (x)^+ = \max(0, x)

    参数：
        - **x** (COOTensor) - coo_relu的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，
          其数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_。

    返回：
        COOTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是数值型。
        - **TypeError** - `x` 不是COOTensor。
