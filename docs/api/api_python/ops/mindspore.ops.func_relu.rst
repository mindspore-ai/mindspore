mindspore.ops.relu
==================

.. py:function:: mindspore.ops.relu(x)

    线性修正单元激活函数（Rectified Linear Unit）。

    返回 :math:`\max(x,\  0)` 的值，负值神经元将被设置为0，正值神经元将保持不变。

    .. math::
        ReLU(x) = (x)^+ = max(0, x)

    .. note::
        一般来说，与 `ReLUV2` 相比，此算子更常用。且 `ReLUV2` 会多输出一个掩码。

    参数：
        - **x** (Tensor) - relu的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，
          其数据类型为 `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_。

    返回：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是数值型。
        - **TypeError** - `x` 不是Tensor。
