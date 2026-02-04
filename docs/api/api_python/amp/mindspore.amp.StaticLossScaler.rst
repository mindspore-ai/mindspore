mindspore.amp.StaticLossScaler
==============================

.. py:class:: mindspore.amp.StaticLossScaler(scale_value)

    Static Loss scale类。使用固定常数对损失或梯度进行缩放和反缩放操作。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **scale_value** (Union(float, int)) - 缩放系数。

    .. py:method:: adjust(grads_finite)

        用于调整 `LossScaler` 中 `scale_value` 的值。`StaticLossScaler` 中，`scale_value` 值固定，因此此方法直接返回False。

        参数：
            - **grads_finite** (Tensor) - bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。

    .. py:method:: scale(inputs)

        对inputs进行缩放，`inputs \*= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

        返回：
            Union(Tensor, tuple(Tensor))，缩放后的值。

    .. py:method:: unscale(inputs)

        对inputs进行反缩放，`inputs /= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

        返回：
            Union(Tensor, tuple(Tensor))，反缩放后的值。
