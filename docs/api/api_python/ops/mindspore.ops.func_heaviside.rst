mindspore.ops.heaviside
========================

.. py:function:: mindspore.ops.heaviside(x, values)

    计算输入中每​​个元素的 Heaviside 阶跃函数。公式定义如下：

    .. math::
            \text { heaviside }(\text { x, values })=\left\{\begin{array}{ll}
            0, & \text { if x }<0 \\
            \text { values, } & \text { if x }==0 \\
            1, & \text { if x }>0
            \end{array}\right.

    参数：
        - **x** (Tensor) - 输入Tensor。需为实数类型。
        - **values** (Tensor) - `x` 为零时填充的值。 `values` 可以被 `x` 广播。 `x` 需要与 `values` 数据类型相同。

    返回：
        Tensor，数据类型与输入 `x` 和 `values` 相同。

    异常：
        - **TypeError** - `x` 或 `values` 不是Tensor。
        - **TypeError** - `x` 和 `values` 的数据类型不同。
        - **ValueError** - 两个输入参数的shape不支持广播。
