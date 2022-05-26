mindspore.ops.ReLU6
====================

.. py:class:: mindspore.ops.ReLU6

    计算输入Tensor的ReLU（矫正线性单元），其上限为6。

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    返回 :math:`\min(\max(0,x), 6)` 元素的值。

    **输入：**

    - **input_x** (Tensor) - ReLU6的输入，任意维度的Tensor，数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `input_x` 相同。

    **异常：**

    - **TypeError** - 如果 `input_x` 的数据类型既不是float16也不是float32。
    - **TypeError** - 如果 `input_x` 不是Tensor。