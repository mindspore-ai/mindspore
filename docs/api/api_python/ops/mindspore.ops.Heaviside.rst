mindspore.ops.Heaviside
=======================

.. py:class:: mindspore.ops.Heaviside

    计算输入中每个元素的Heaviside步长函数。

    .. math::
            \text { heaviside }(\text { x, values })=\left\{\begin{array}{ll}
            0, & \text { if x }<0 \\
            \text { values, } & \text { if x }==0 \\
            1, & \text { if x }>0
            \end{array}\right.

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x** (Tensor) - 输入Tensor，实数类型。
        - **values** (Tensor) - 在 `x` 中为0的位置应用其值。 `values` 可以同 `x` 进行广播。 `x` 与 `values` 的数据类型应该相同。

    输出：
        Tensor，与 `x` 和 `values` 的数据类型相同。

    异常：
        - **TypeError** - 如果 `x` 或 `values` 不是Tensor。
        - **TypeError** - 如果 `x` 与 `values` 的数据类型不一致。
        - **ValueError** - 如果两个输入的shape之间无法进行广播。
