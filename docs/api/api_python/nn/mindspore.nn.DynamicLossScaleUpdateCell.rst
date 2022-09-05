mindspore.nn.DynamicLossScaleUpdateCell
=======================================

.. py:class:: mindspore.nn.DynamicLossScaleUpdateCell(loss_scale_value, scale_factor, scale_window)

    用于动态更新损失缩放系数(loss scale)的神经元。

    使用混合精度功能进行训练时，初始损失缩放系数值为 `loss_scale_value`。在每个训练步骤中，当出现溢出时，通过计算公式 `loss_scale`/`scale_factor` 减小损失缩放系数。如果连续 `scale_window` 步（step）未溢出，则将通过 `loss_scale` * `scale_factor` 增大损失缩放系数。

    该类是 :class:`mindspore.amp.DynamicLossScaleManager` 的 `get_update_cell` 方法的返回值。训练过程中，类 :class:`mindspore.nn.TrainOneStepWithLossScaleCell` 会调用该Cell来更新损失缩放系数。

    参数：
        - **loss_scale_value** (float) - 初始的损失缩放系数。
        - **scale_factor** (int) - 增减系数。
        - **scale_window** (int) - 未溢出时，增大损失缩放系数的最大连续训练步数。

    输入：
        - **loss_scale** (Tensor) - 训练期间的损失缩放系数，是一个标量。
        - **overflow** (bool) - 是否发生溢出。

    输出：
        Bool，即输入 `overflow` 。

    .. py:method:: get_loss_scale()

        获取当前损失缩放系数。

        返回：
            float，损失缩放系数。