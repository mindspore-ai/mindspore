mindspore.amp.LossScaler
========================

.. py:class:: mindspore.amp.LossScaler

    使用混合精度时，用于管理损失缩放系数（loss scaler）的抽象类。

    派生类需要实现该类的所有方法。训练过程中，`scale` 和 `unscale` 用于对损失值或梯度进行放大或缩小，以避免数据溢出；`adjust` 用于调整损失缩放系数 `scale_value` 的值。


    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. py:method:: adjust(grads_finite)
        :abstractmethod:

        根据梯度是否为有效值（无溢出）对 `scale_value` 进行调整。

        参数：
            - **grads_finite** (Tensor) - bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。

    .. py:method:: scale(inputs)
        :abstractmethod:

        对inputs进行缩放，`inputs \*= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

    .. py:method:: unscale(inputs)
        :abstractmethod:

        对inputs进行反缩放，`inputs /= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。
