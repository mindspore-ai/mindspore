mindspore.ops.heaviside
========================

.. py:function:: mindspore.ops.heaviside(input, values)

    计算输入中每个元素的 Heaviside 阶跃函数。公式定义如下：

    .. math::
            \text { heaviside }(\text { input, values })=\left\{\begin{array}{ll}
            0, & \text { if input }<0 \\
            \text { values, } & \text { if input }==0 \\
            1, & \text { if input }>0
            \end{array}\right.

    参数：
        - **input** (Tensor) - 输入Tensor。需为实数类型。
        - **values** (Tensor) - `input` 为零时填充的值。 `values` 可以被 `input` 广播。 `input` 需要与 `values` 数据类型相同。

    返回：
        Tensor，数据类型与输入 `input` 和 `values` 相同。

    异常：
        - **TypeError** - `input` 或 `values` 不是Tensor。
        - **TypeError** - `input` 和 `values` 的数据类型不同。
        - **ValueError** - 两个输入参数的shape不支持广播。
