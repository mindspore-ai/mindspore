.. py:class:: mindspore.train.callback.LossMonitor(per_print_times=1)

    监控训练的loss。

    如果loss是NAN或INF，则终止训练。

    .. note::
        如果 `per_print_times` 为0，则不打印loss。

    **参数：**

    - **per_print_times** (int) - 表示每隔多少个step打印一次loss。默认值：1。

    **异常：**

    - **ValueError** - 当 `per_print_times` 不是整数或小于零。

    **样例：**

    >>> from mindspore import Model, nn
    >>>
    >>> net = LeNet5()
    >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    >>> model = Model(net, loss_fn=loss, optimizer=optim)
    >>> data_path = './MNIST_Data'
    >>> dataset = create_dataset(data_path)
    >>> time_monitor = TimeMonitor()
    >>> model.train(10, dataset, callbacks=time_monitor)


    .. py:method:: step_end(run_context)

        step结束时打印训练loss。

        **参数：**

        - **run_context** (RunContext) - 包含模型的相关信息。
