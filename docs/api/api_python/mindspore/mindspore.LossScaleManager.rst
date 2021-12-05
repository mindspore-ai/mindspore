mindspore.LossScaleManager
===========================

.. py:class:: mindspore.LossScaleManager

    混合精度梯度放大系数（loss scale）管理器的抽象类。

    派生类需要实现该类的所有方法。 `get_loss_scale` 用于获取当前的梯度放大系数。 `update_loss_scale` 用于更新梯度放大系数，该方法将在训练过程中被调用。 `get_update_cell` 用于获取更新梯度放大系数的 `Cell` 实例，该实例将在训练过程中被调用。当前多使用 `get_update_cell` 方式。

    例如：:class:`mindspore.FixedLossScaleManager` 和 :class:`mindspore.DynamicLossScaleManager` 。

    .. py:method:: get_loss_scale()

        获取梯度放大系数（loss scale）的值。

    .. py:method:: get_update_cell()

        获取用于更新梯度放大系数的Cell实例。

    .. py:method:: update_loss_scale(overflow)

        根据 `overflow` 状态更新梯度放大系数（loss scale)。

        **参数：**

        - **overflow** (bool) - 表示训练过程是否溢出。