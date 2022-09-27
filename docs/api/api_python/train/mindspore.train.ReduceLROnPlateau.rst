mindspore.train.ReduceLROnPlateau
=================================

.. py:class:: mindspore.train.ReduceLROnPlateau(monitor='eval_loss', factor=0.1, patience=10, verbose=False, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0)

    当 `monitor` 停止改进时降低学习率。

    一旦学习停止，模型通常受益于降低2-10倍的学习率。此回调监控训练过程，当在 `patience` 个epoch范围内指标效果没有变好时，学习率就会降低。

    .. note::
        暂不支持分组学习率场景。

    参数：
        - **monitor** (str) - 监控指标。如果是边训练边推理场景，合法的monitor配置值可以为"loss", "eval_loss"以及实例化 `Model` 时传入的metric名称；如果在训练时不做推理，合法的monitor配置值为"loss"。当monitor为"loss"时，如果训练网络有多个输出，默认取第一个值为训练损失值。默认值："eval_loss"。
        - **factor** (float) - 学习率变化系数，范围在0-1之间。默认值：0.1。
        - **patience** (int) - `moniter` 相对历史最优值变好超过 `min_delta` 视为当前epoch的模型效果有所改善，`patience` 为等待的无改善epoch的数量，当内部等待的epoch数 `self.wait` 大于等于 `patience` 时，训练停止。默认值：10。
        - **verbose** (bool) - 是否打印相关信息。默认值：False。
        - **mode** (str) - `{'auto', 'min', 'max'}` 中的一种，'min'模式下将在指标不再减小时改变学习率，'max'模式下将在指标不再增大时改变学习率，'auto'模式将根据当前 `monitor` 指标的特点自动设置。默认值："auto"。
        - **min_delta** (float) - `monitor` 指标变化的最小阈值，超过此阈值才视为 `monitor` 的变化。默认值：1e-4。
        - **cooldown** (int) - 减小学习率后，在接下来的 `cooldown` 个epoch中不执行操作。默认值：0。
        - **min_lr** (float) - 学习率最小设定值。默认值：0。

    异常：
        - **ValueError** - 当 `mode` 不在 `{'auto', 'min', 'max'}` 中。
        - **ValueError** - 分组学习率或动态学习率场景下，当获取到的学习率不是parameter类型。
        - **ValueError** - 当传入的 `monitor` 返回值不是标量。

    .. py:method:: on_train_begin(run_context)

        训练开始时初始化相关的变量。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_epoch_end(run_context)

        训练过程中，若监控指标在等待 `patience` 个epoch后仍没有改善，则改变学习率。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。
