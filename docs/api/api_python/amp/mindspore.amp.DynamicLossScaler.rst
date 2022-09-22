mindspore.amp.DynamicLossScaler
===============================

.. py:class:: mindspore.amp.DynamicLossScaler(scale_value, scale_factor, scale_window)

    动态调整损失缩放系数的管理器。

    动态损失缩放管理器在保证梯度不溢出的情况下，尝试确定最大的损失缩放值 `scale_value`。在梯度不溢出的情况下，`scale_value` 将会每间隔 `scale_window` 步被扩大 `scale_factor` 倍，若存在溢出情况，则会将 `scale_value` 缩小 `scale_factor` 倍，并重置计数器。

    .. note::
        - 这是一个实验性接口，后续可能删除或修改。

    参数：
        - **scale_value** (Union(float, int)) - 初始梯度放大系数。
        - **scale_factor** (int) - 放大/缩小倍数。
        - **scale_window** (int) - 无溢出时的连续正常step的最大数量。

    .. py:method:: adjust(grads_finite)

        根据梯度是否为有效值（无溢出）对 `scale_value` 进行调整。

        参数：
            - **grads_finite** (Tensor) - bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。

    .. py:method:: scale(inputs)

        根据 `scale_value` 放大inputs。

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
