mindspore.Tensor.floor_divide\_
===============================

.. py:method:: mindspore.Tensor.floor_divide_(other) -> Tensor

    按元素将 `self` Tensor除以 `other` Tensor，并向下取整。

    .. math::
        out_{i} = \text{floor}( \frac{self_i}{other_i})

    其中 :math:`floor` 表示Floor算子。有关更多详细信息，请参阅 :class:`mindspore.mint.floor` 算子。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        当 `self` 和 `other` 具有不同的shape时， `other` 必须能被广播成 `self` 。

    参数：
        - **other** (Union[Tensor, Number, bool]) - 第二个输入，为数值型，或bool，或数据类型为数值型或bool的Tensor。

    返回：
        Tensor，shape与 `self` 的shape相同，数据类型和 `self` 的数据类型相同。

    异常：
        - **TypeError** - 如果 `other` 不是以下之一：Tensor，number.Number或bool。
        - **RuntimeError** - 如果 `other` 不能被广播成 `self` 。
