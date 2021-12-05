.. py:class:: mindspore.train.callback.LearningRateScheduler(learning_rate_function)

    在训练期间更改学习率。

    **参数：**

    - **learning_rate_function** (Function) - 在训练期间更改学习率的函数。

    **样例：**

    >>> from mindspore import Model
    >>> from mindspore.train.callback import LearningRateScheduler
    >>> import mindspore.nn as nn
    ...
    >>> def learning_rate_function(lr, cur_step_num):
    ...     if cur_step_num%1000 == 0:
    ...         lr = lr*0.1
    ...     return lr
    ...
    >>> lr = 0.1
    >>> momentum = 0.9
    >>> net = Net()
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum)
    >>> model = Model(net, loss_fn=loss, optimizer=optim)
    ...
    >>> dataset = create_custom_dataset("custom_dataset_path")
    >>> model.train(1, dataset, callbacks=[LearningRateScheduler(learning_rate_function)],
    ...             dataset_sink_mode=False)

    .. py:method:: step_end(run_context)

        在step结束时更改学习率。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。
