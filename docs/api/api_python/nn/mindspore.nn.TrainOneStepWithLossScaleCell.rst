mindspore.nn.TrainOneStepWithLossScaleCell
==========================================

.. py:class:: mindspore.nn.TrainOneStepWithLossScaleCell(network, optimizer, scale_sense)

    使用混合精度功能的训练网络。

    实现了包含损失缩放（loss scale）的单次训练。它使用网络、优化器和用于更新损失缩放系数（loss scale）的Cell(或一个Tensor)作为参数。可在host侧或device侧更新损失缩放系数。
    如果需要在host侧更新，使用Tensor作为 `scale_sense` ，否则，使用可更新损失缩放系数的Cell实例作为 `scale_sense` 。

    参数：
        - **network** (Cell) - 训练网络。仅支持单输出网络。
        - **optimizer** (Cell) - 用于更新网络参数的优化器。
        - **scale_sense** (Union[Tensor, Cell]) - 如果此值为Cell类型，`TrainOneStepWithLossScaleCell` 会调用它来更新损失缩放系数。如果此值为Tensor类型，可调用 `set_sense_scale` 来更新损失缩放系数，shape为 :math:`()` 或 :math:`(1,)` 。

    输入：
        - **\*inputs** (Tuple(Tensor)) - shape为 :math:`(N, \ldots)` 的Tensor组成的元组。

    输出：
        Tuple，包含三个Tensor，分别为损失函数值、溢出状态和当前损失缩放系数。

        - **loss** （Tensor） - 标量，表示损失函数值。
        - **overflow** （Tensor）- 类型为bool的标量，表示是否发生溢出。
        - **loss scale** （Tensor）- 表示损失放大系数，shape为 :math:`()` 或 :math:`(1,)` 。

    异常：
        - **TypeError** - `scale_sense` 既不是Cell，也不是Tensor。
        - **ValueError** - `scale_sense` 的shape既不是(1,)也不是()。

    .. py:method:: get_overflow_status(status, compute_output)

        获取浮点溢出状态。

        溢出检测的目标过程执行完成后，获取溢出结果。继承该类自定义训练网络时，可复用该接口。

        参数：
            - **status** (object) - 用于控制与 `start_overflow_check` 的执行序，应设置为 `start_overflow_check` 的第一输出。
            - **compute_output** - 对特定计算过程进行溢出检测时，将 `compute_output` 设置为该计算过程的输出。

        返回：
            bool，是否发生溢出。

    .. py:method:: process_loss_scale(overflow)

        根据溢出状态计算损失缩放系数。
        
        继承该类自定义训练网络时，可复用该接口。

        参数：
            - **overflow** (bool) - 是否发生溢出。

        返回：
            bool，溢出状态，即输入。

    .. py:method:: set_sense_scale(sens)

        如果使用了Tensor类型的 `scale_sense` ，可调用此函数修改它的值。

        参数：
            - **sens** (Tensor) - 新的损失缩放系数，其shape和类型需要与原始 `scale_sense` 相同。

    .. py:method:: start_overflow_check(pre_cond, compute_input)

        启动浮点溢出检测。创建并清除溢出检测状态。

        指定参数 `pre_cond` 和 `compute_input` ，以确保在正确的时间清除溢出状态。以当前接口为例，我们需要在损失函数计算后进行清除状态，在梯度计算过程中检测溢出。在这种情况下，`pre_cond` 应为损失函数的输出，而 `compute_input` 应为梯度计算函数的输入。继承该类自定义训练网络时，可复用该接口。

        参数：
            - **pre_cond** (Tensor) - 启动溢出检测的先决条件。它决定溢出状态清除和先前处理的执行顺序。它确保函数 `start_overflow` 在执行完先决条件后清除状态。
            - **compute_input** (object) - 后续运算的输入。需要对特定的计算过程进行溢出检测。将 `compute_input` 设置这一计算过程的输入，以确保在执行该计算之前清除了溢出状态。

        返回：
            Tuple[object, object]，第一输出用于控制执行序，为保证编译优化后 `start_overflow_check` 在 `get_overflow_status` 前执行，该值应作为 `get_overflow_status` 的第一个输入。第二输出与 `compute_input` 的输入相同，用于控制执行序，保证在函数返回时完成对溢出标志的清理。
