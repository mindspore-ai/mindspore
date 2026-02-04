mindspore.amp.LossScaleManager
==============================

.. py:class:: mindspore.amp.LossScaleManager

    使用混合精度时，用于管理损失缩放系数（loss scale）的抽象类。

    派生类需要实现该类的所有方法。 `get_loss_scale` 用于获取当前的梯度放大系数。 `update_loss_scale` 用于更新梯度放大系数，该方法将在训练过程中被调用。 `get_update_cell` 用于获取更新梯度放大系数的 :class:`mindspore.nn.Cell` 实例，该实例将在训练过程中被调用。当前通常使用 `get_update_cell` 方法。

    例如：:class:`mindspore.amp.FixedLossScaleManager` 和 :class:`mindspore.amp.DynamicLossScaleManager` 。

    .. py:method:: get_loss_scale()

        获取梯度放大系数（loss scale）的值。

    .. py:method:: get_update_cell()

        获取用于更新梯度放大系数的 :class:`mindspore.nn.Cell` 实例。

    .. py:method:: update_loss_scale(overflow)

        根据 `overflow` 状态更新梯度放大系数（loss scale）。

        参数：
            - **overflow** (bool) - 表示训练过程是否溢出。
