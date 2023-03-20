mindspore.amp.LossScaler
========================

.. py:class:: mindspore.amp.LossScaler

    使用混合精度时，用于管理损失缩放系数（loss scaler）的抽象类。

    派生类需要实现该类的所有方法。训练过程中，`scale` 和 `unscale` 用于对损失值或梯度进行放大或缩小，以避免数据溢出；`adjust` 用于调整损失缩放系数 `scale_value` 的值。

    关于使用 `LossScaler` 进行损失缩放，请查看 `教程 <https://mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html#%E6%8D%9F%E5%A4%B1%E7%BC%A9%E6%94%BE>`_。

    .. note::
        - 这是一个实验性接口，后续可能删除或修改。

    .. py:method:: adjust(grads_finite)

        根据梯度是否为有效值（无溢出）对 `scale_value` 进行调整。

        参数：
            - **grads_finite** (Tensor) - bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。

    .. py:method:: scale(inputs)

        对inputs进行scale，`inputs \*= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。

    .. py:method:: unscale(inputs)

        对inputs进行unscale，`inputs /= scale_value`。

        参数：
            - **inputs** (Union(Tensor, tuple(Tensor))) - 损失值或梯度。
