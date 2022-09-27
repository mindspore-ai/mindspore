mindspore.ops.assign_sub
========================

.. py:function:: mindspore.ops.assign_sub(variable, value)

    从网络参数减去特定数值来更新网络参数。

    输入 `variable` 和 `value` 会通过隐式数据类型转换使数据类型保持一致。如果数据类型不同，低精度的数据类型会被转换到高精度的数据类型。如果 `value` 为标量会被自动转换为Tensor，其数据类型会与 `variable` 保持一致。

    .. Note::
        由于 `variable` 类型为 `Parameter` ，其数据类型不能改变，因此只允许 `value` 的数据类型转变为 `variable` 的数据类型。而且由于不同设备支持的转换类型会有所不同，推荐在使用此操作时使用相同的数据类型。

    参数：
        - **variable** (Parameter) - 待更新的网络参数，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其轶应小于8。
        - **value** (Union[numbers.Number, Tensor]) - 从 `variable` 减去的值。如果类型为Tensor则应与 `variable` 的shape相同。使用此操作时推荐使用相同的数据类型。

    返回：
        Tensor，shape和数据类型与 `variable` 相同。

    异常：
        - **TypeError** - `value` 不是标量或Tensor。
        - **RuntimeError** - `variable` 与 `value` 之间的类型转换不被支持。
