mindspore.amp.StaticLossScaler
==============================

.. py:class:: mindspore.amp.StaticLossScaler(scale_value)

    Static Loss scale类。用固定的常数来scales和unscale损失或梯度。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **scale_value** (Union(float, int)) - 缩放系数。

    .. py:method:: adjust(grads_finite)

        用于调整 `LossScaler` 中 `loss_value` 的值。`StaticLossScaler` 中，`scale_value` 值固定，因此此方法直接返回False。

        参数：
            - **grads_finite** (Tensor) - bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。

    .. py:method:: scale(inputs)

        对inputs进行scale，`inputs \*= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

        返回：
            Union(Tensor, tuple(Tensor))，scale后的值。

    .. py:method:: unscale(inputs)

        对inputs进行unscale，`inputs /= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

        返回：
            Union(Tensor, tuple(Tensor))，unscale后的值。
