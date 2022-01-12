mindspore.FixedLossScaleManager
===============================

.. py:class:: mindspore.FixedLossScaleManager(loss_scale=128.0, drop_overflow_update=True)

    损失缩放系数不变的管理器，继承自 :class:`mindspore.LossScaleManager` 。

    **参数：**

    - **loss_scale** (float) - 梯度放大系数。注：如果将 `drop_overflow_update` 设为False，则定义优化器时需要将优化器的 `loss_scale` 设为相同的值。默认值：128.0。
    - **drop_overflow_update** (bool) - 出现溢出时，是否执行优化器。如果值为True，则出现溢出时不会执行优化器。默认值：True。

    **样例：**

    >>> from mindspore import Model, nn, FixedLossScaleManager
    >>>
    >>> net = Net()
    >>> # 1) 如果存在溢出，则不执行参数更新
    >>> loss_scale_manager = FixedLossScaleManager()
    >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    >>> model = Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)
    >>>
    >>> # 2) 即使发生溢出，也执行参数更新
    >>> loss_scale = 1024.0
    >>> loss_scale_manager = FixedLossScaleManager(loss_scale, False)
    >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9, loss_scale=loss_scale)
    >>> model = Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)

    .. py:method:: get_drop_overflow_update()

        返回 `drop_overflow_update` ，该值表示是否在发生溢出时放弃本轮参数更新。

        **返回：**

        bool, `drop_overflow_update` 的值。

    .. py:method:: get_loss_scale()

        获取loss scale值。

        **返回：**

        bool，`loss_scale` 的值。

    .. py:method:: get_update_cell()

        返回用于更新 `loss_scale` 值的 `Cell` 实例， :class:`mindspore.TrainOneStepWithLossScaleCell` 会调用该实例。该类使用固定的梯度放大系数，因此该实例不执行任何操作。

        **返回：**

        None或 `Cell` 。当 `drop_overflow_update` 为True时，返回 :class:`mindspore.FixedLossScaleUpdateCell` 实例，当 `drop_overflow_update` 为False时，返回None。

    .. py:method:: update_loss_scale(overflow)

        更新loss scale值。类 :class:`mindspore.FixedLossScaleManager` 中，该方法不执行任何操作。

        **参数：**

        - **overflow** (bool) - 表示是否溢出。
