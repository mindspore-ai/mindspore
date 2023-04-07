mindspore.nn.FixedLossScaleUpdateCell
=======================================

.. py:class:: mindspore.nn.FixedLossScaleUpdateCell(loss_scale_value)

    固定损失缩放系数的神经元。

    该类是 :class:`mindspore.amp.FixedLossScaleManager` 的 `get_update_cell` 方法的返回值。训练过程中，类 :class:`mindspore.nn.TrainOneStepWithLossScaleCell` 会调用该Cell。

    参数：
        - **loss_scale_value** (float) - 初始损失缩放系数。

    输入：
        - **loss_scale** (Tensor) - 训练期间的损失缩放系数，是一个标量，shape为 :math:`()`。在当前类中，该值被忽略。
        - **overflow** (bool) - 是否发生溢出。

    输出：
        Bool，即输入 `overflow`。

    .. py:method:: get_loss_scale()

        获取当前损失缩放系数。

        返回：
            float，损失缩放系数。