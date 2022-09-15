mindspore.ops.relu6
====================

.. py:function:: mindspore.ops.relu6(input_x)

    计算输入Tensor的ReLU（修正线性单元），其上限为6。

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    返回 :math:`\min(\max(0,x), 6)` 元素的值。

    参数：
        - **input_x** (Tensor) - relu6的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，数据类型为float16或float32。

    返回：
        Tensor，数据类型和shape与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `input_x` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `input_x` 不是Tensor。
