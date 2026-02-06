mindspore.amp.DynamicLossScaleManager
=====================================

.. py:class:: mindspore.amp.DynamicLossScaleManager(init_loss_scale=2 ** 24, scale_factor=2, scale_window=2000)

    动态调整损失缩放系数的管理器，继承自 :class:`mindspore.amp.LossScaleManager` 。

    参数：
        - **init_loss_scale** (float，可选) - 初始梯度放大系数。默认值： ``2 ** 24`` 。
        - **scale_factor** (int，可选) - 放大/缩小倍数。默认值： ``2`` 。
        - **scale_window** (int，可选) - 无溢出时的连续正常step的最大数量。默认值： ``2000`` 。

    .. py:method:: get_drop_overflow_update()

        该值表示是否在发生溢出时放弃本轮参数更新。

        返回：
            bool，值始终为 ``True``。

    .. py:method:: get_loss_scale()

        返回当前梯度放大系数。

        返回：
            float，梯度放大系数。

    .. py:method:: get_update_cell()

        返回用于更新梯度放大系数的 :class:`mindspore.nn.Cell` 实例。:class:`mindspore.nn.TrainOneStepWithLossScaleCell` 会调用该实例。

        返回：
            :class:`mindspore.nn.DynamicLossScaleUpdateCell` 实例，用于更新梯度放大系数。

    .. py:method:: update_loss_scale(overflow)

        根据溢出状态更新梯度放大系数。如果发生溢出，减小梯度放大系数；否则增大梯度放大系数。

        参数：
            - **overflow** (bool) - 表示是否发生溢出。
