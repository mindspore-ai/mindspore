.. py:class:: mindspore.train.callback.Callback

    用于构建回调函数的基类。回调函数是一个上下文管理器，在运行模型时被调用。
    可以使用此机制进行初始化和释放资源等操作。

    回调函数可以在step或epoch中的执行一些操作。

    它保存模型相关信息。例如 `network` 、 `train_network` 、 `epoch_num` 、 `batch_num` 、 `loss_fn` 、 `optimizer` 、 `parallel_mode` 、 `device_number` 、 `list_callback` 、 `cur_epoch_num` 、 `cur_step_num` 、 `dataset_sink_mode` 、 `net_outputs` 等。

    **样例：**

    >>> from mindspore import Model, nn
    >>> from mindspore.train.callback import Callback
    >>> class Print_info(Callback):
    ...     def step_end(self, run_context):
    ...         cb_params = run_context.original_args()
    ...         print("step_num: ", cb_params.cur_step_num)
    >>>
    >>> print_cb = Print_info()
    >>> dataset = create_custom_dataset()
    >>> net = Net()
    >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    >>> model = Model(net, loss_fn=loss, optimizer=optim)
    >>> model.train(1, dataset, callbacks=print_cb)
    step_num：1

    .. py:method:: begin(run_context)

        在网络执行之前被调用一次。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: end(run_context)

        网络执行后被调用一次。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: epoch_begin(run_context)

        在每个epoch开始之前被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: epoch_end(run_context)

        在每个epoch结束后被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: step_begin(run_context)

        在每个step开始之前被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: step_end(run_context)

        在每个step完成后被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。
