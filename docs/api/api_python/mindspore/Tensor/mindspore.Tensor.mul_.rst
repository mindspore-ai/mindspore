mindspore.Tensor.mul\_
==========================

.. py:method:: mindspore.Tensor.mul_(other) -> Tensor

    两个Tensor逐元素相乘。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. math::

        out_{i} = tensor_{i} * other_{i}

    .. note::
        - 当两个输入具有不同的shape时， `other` 必须能被广播成 `self` 。
        - `self` 和 `other` 不能同时为bool类型。[True, Tensor(True), Tensor(np.array([True]))]等都为bool类型。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - `other` 可以是number.Number或bool，也可以是数据类型为number.Number或bool的Tensor。

    返回：
        Tensor，shape与 `self` 的shape相同，数据类型和 `self` 的类型相同。

    异常：
        - **TypeError** - `other` 不是Tensor、number.Number或bool。
        - **RuntimeError** - `other` 不能被广播成 `self`。
