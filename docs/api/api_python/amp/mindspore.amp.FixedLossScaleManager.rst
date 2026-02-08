mindspore.amp.FixedLossScaleManager
===================================

.. py:class:: mindspore.amp.FixedLossScaleManager(loss_scale=128.0, drop_overflow_update=True)

    损失缩放系数不变的管理器，继承自 :class:`mindspore.amp.LossScaleManager` 。

    参数：
        - **loss_scale** (float，可选) - 梯度放大系数。注：如果将 `drop_overflow_update` 设为 ``False`` ，则定义优化器时需要将优化器的 `loss_scale` 设为相同的值。默认值： ``128.0`` 。
        - **drop_overflow_update** (bool，可选) - 出现溢出时，是否执行优化器。如果值为 ``True`` ，则出现溢出时不会执行优化器。默认值： ``True`` 。

    .. py:method:: get_drop_overflow_update()

        返回 `drop_overflow_update` ，该值表示是否在发生溢出时放弃本轮参数更新。

        返回：
            bool， `drop_overflow_update` 的值。

    .. py:method:: get_loss_scale()

        获取loss scale值。

        返回：
            float，`loss_scale` 的值。

    .. py:method:: get_update_cell()

        返回用于更新 `loss_scale` 值的 :class:`mindspore.nn.Cell` 实例。 :class:`mindspore.nn.TrainOneStepWithLossScaleCell` 会调用该实例。该类使用固定的梯度放大系数，因此该实例不执行任何操作。

        返回：
            None或 `Cell` 。当 `drop_overflow_update` 为 ``True`` 时，返回 :class:`mindspore.nn.FixedLossScaleUpdateCell` 实例；当 `drop_overflow_update` 为 ``False`` 时，返回None。

    .. py:method:: update_loss_scale(overflow)

        更新loss scale值。在 :class:`mindspore.amp.FixedLossScaleManager` 类中，该方法不执行任何操作。

        参数：
            - **overflow** (bool) - 表示是否溢出。
