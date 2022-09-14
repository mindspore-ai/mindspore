mindspore.amp.StaticLossScaler
==============================

.. py:class:: mindspore.amp.StaticLossScaler(scale_value)

    Static Loss scale类。用固定的常数来scales和unscale损失或梯度。

    .. note::
        - 这是一个实验性接口，后续可能删除或修改。

    参数：
        - **scale_value** (Union(float, int)) - 缩放系数。

    .. py:method:: adjust(grads_finite)

        `scale_value` 值固定。

        参数：
            - **grads_finite** (Tensor) - bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。

    .. py:method:: scale(inputs)

        对inputs进行scale，`inputs \*= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

        返回：
            Union(Tensor, tuple(Tensor))，缩放的值。

    .. py:method:: unscale(inputs)

        对inputs进行unscale，`inputs /= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

        返回：
            Union(Tensor, tuple(Tensor))，未缩放的值。
