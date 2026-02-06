mindspore.train.EarlyStopping
=============================

.. py:class:: mindspore.train.EarlyStopping(monitor='eval_loss', min_delta=0, patience=0, verbose=False, mode='auto', baseline=None, restore_best_weights=False)

    当监控的指标停止改进时，停止训练。

    假设 `monitor` 是 ``'accuracy'``，那么，`mode` 将为 ``'max'``，因为训练的目标是准确率的提高。 `model.fit()` 边训练边验证场景下，将记录 `monitor` 的变化。当在 `patience` 个epoch范围内指标效果变好的程度没有超过 `min_delta` 时，将调用 `run_context.request_stop()` 方法来终止训练。

    参数：
        - **monitor** (str，可选) - 监控指标。如果是边训练边推理场景，合法的 `monitor` 配置值可以为 ``'loss'``、 ``'eval_loss'`` 以及实例化 `Model` 时传入的metric名称；如果在训练时不做推理，合法的 `monitor` 配置值为 ``'loss'``。当 `monitor` 为 ``'loss'`` 时，如果训练网络有多个输出，默认取第一个值为训练损失值。默认值： ``'eval_loss'`` 。
        - **min_delta** (float，可选) - `monitor` 指标变化的最小阈值，超过此阈值才视为 `monitor` 的变化。默认值： ``0`` 。
        - **patience** (int，可选) - `monitor` 相对历史最优值变好超过 `min_delta` ，视为当前epoch的模型效果有所改善。`patience` 为等待的无改善epoch的数量。当内部等待的epoch数 `self.wait` 大于等于 `patience` 时，训练停止。默认值： ``0`` 。
        - **verbose** (bool，可选) - 是否打印相关信息。默认值： ``False`` 。
        - **mode** (str，可选) - `{'auto', 'min', 'max'}` 中的一种。默认值： ``'auto'`` 。

          - ``'min'`` 模式下，将在指标不再减小时，执行早停。
          - ``'max'`` 模式下，将在指标不再增大时，执行早停。
          - ``'auto'`` 模式下，将根据当前 `monitor` 指标的特点自动设置。

        - **baseline** (float，可选) - 模型效果的基线。当前 `monitor` 相对历史最优值变好，且好于 `baseline` 时，内部的等待epoch计数器被清零。默认值： ``None`` 。
        - **restore_best_weights** (bool，可选) - 是否自动保存最优模型的权重。默认值： ``False`` 。

    异常：
        - **ValueError** - 当 `mode` 不在 `{'auto', 'min', 'max'}` 中。
        - **ValueError** - 当传入的 `monitor` 返回值不是标量。

    .. py:method:: on_train_begin(run_context)

        训练开始时初始化相关的变量。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_end(run_context)

        打印在第几个epoch执行早停。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_epoch_end(run_context)

        训练过程中，若监控指标在等待 `patience` 个epoch后仍没有改善，则停止训练。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

