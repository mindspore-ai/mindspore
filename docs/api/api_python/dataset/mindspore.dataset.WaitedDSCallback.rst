mindspore.dataset.WaitedDSCallback
==================================

.. py:class:: mindspore.dataset.WaitedDSCallback(step_size=1)

    阻塞式数据处理回调类的抽象基类，用于与训练回调类(`mindspore.callback <https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.train.html#mindspore.train.callback.Callback>`_)的同步。

    可用于在step或epoch开始前执行自定义的回调方法，例如在自动数据增强中根据上一个epoch的loss值来更新增强算子参数配置。

    注意，第2个step或epoch开始时才会触发该调用。

    用户可通过 `train_run_context` 获取模型相关信息，如 `network` 、 `train_network` 、 `epoch_num` 、 `batch_num` 、 `loss_fn` 、 `optimizer` 、 `parallel_mode` 、 `device_number` 、 `list_callback` 、 `cur_epoch_num` 、 `cur_step_num` 、 `dataset_sink_mode` 、 `net_outputs` 等，详见 `mindspore.callback <https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.train.html#mindspore.train.callback.Callback>`_ 。

    用户可通过 `ds_run_context` 获取数据处理管道相关信息，包括 `cur_epoch_num` (当前epoch数)、 `cur_step_num_in_epoch` (当前epoch的step数)、 `cur_step_num` (当前step数)。

    **参数：**

    - **step_size** (int, optional) - 每个step包含的数据行数。通常step_size与batch_size一致，默认值：1。

    **样例：**

    >>> import mindspore.nn as nn
    >>> from mindspore.dataset import WaitedDSCallback
    >>> from mindspore import context
    >>> from mindspore.train import Model
    >>> from mindspore.train.callback import Callback
    >>>
    >>> context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    >>>
    >>> # 自定义用于数据处理管道同步数据的回调类
    >>> class MyWaitedCallback(WaitedDSCallback):
    ...     def __init__(self, events, step_size=1):
    ...         super().__init__(step_size)
    ...         self.events = events
    ...
    ...     # epoch开始前数据处理管道要执行的回调函数
    ...     def sync_epoch_begin(self, train_run_context, ds_run_context):
    ...         event = f"ds_epoch_begin_{ds_run_context.cur_epoch_num}_{ds_run_context.cur_step_num}"
    ...         self.events.append(event)
    ...
    ...     # step开始前数据处理管道要执行的回调函数
    ...     def sync_step_begin(self, train_run_context, ds_run_context):
    ...         event = f"ds_step_begin_{ds_run_context.cur_epoch_num}_{ds_run_context.cur_step_num}"
    ...         self.events.append(event)
    >>>
    >>> # 自定义用于网络训练时同步数据的回调类
    >>> class MyMSCallback(Callback):
    ...     def __init__(self, events):
    ...         self.events = events
    ...
    ...     # epoch结束网络训练要执行的回调函数
    ...     def epoch_end(self, run_context):
    ...         cb_params = run_context.original_args()
    ...         event = f"ms_epoch_end_{cb_params.cur_epoch_num}_{cb_params.cur_step_num}"
    ...         self.events.append(event)
    ...
    ...     # step结束网络训练要执行的回调函数
    ...     def step_end(self, run_context):
    ...         cb_params = run_context.original_args()
    ...         event = f"ms_step_end_{cb_params.cur_epoch_num}_{cb_params.cur_step_num}"
    ...         self.events.append(event)
    >>>
    >>> # 自定义网络
    >>> class Net(nn.Cell):
    ...     def construct(self, x, y):
    ...         return x
    >>>
    >>> # 声明一个网络训练与数据处理同步的数据
    >>> events = []
    >>>
    >>> # 声明数据处理管道和网络训练的回调类
    >>> my_cb1 = MyWaitedCallback(events, 1)
    >>> my_cb2 = MyMSCallback(events)
    >>> arr = [1, 2, 3, 4]
    >>> # 构建数据处理管道
    >>> data = ds.NumpySlicesDataset((arr, arr), column_names=["c1", "c2"], shuffle=False)
    >>> # 将数据处理管道的回调类加入到map中
    >>> data = data.map(operations=(lambda x: x), callbacks=my_cb1)
    >>>
    >>> net = Net()
    >>> model = Model(net)
    >>>
    >>> # 将数据处理管道和网络训练的回调类加入到模型训练的回调列表中
    >>> model.train(2, data, dataset_sink_mode=False, callbacks=[my_cb2, my_cb1])

    .. py:method:: begin(run_context)

        用于定义在网络训练开始前执行的回调方法。

        **参数：**

        - **run_context** (RunContext) - 网络训练运行信息。

    .. py:method:: ds_begin(ds_run_context)

        用于定义在数据处理管道启动前执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_epoch_begin(ds_run_context)

        内部方法，不能被调用或者重写。通过重写mindspore.dataset.DSCallback.ds_epoch_begin 实现与mindspore.train.callback.Callback.epoch_end回调同步。

        **参数：**

        **ds_run_context**：数据处理管道运行信息。

    .. py:method:: ds_epoch_end(ds_run_context)

        用于定义在每个数据epoch结束后执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_step_begin(ds_run_context)

        内部方法，不能被调用或者重写。通过重写mindspore.dataset.DSCallback.ds_step_begin
        实现与mindspore.train.callback.Callback.step_end回调同步。

        **参数：**

        **ds_run_context**：数据处理管道运行信息。

    .. py:method:: ds_step_end(ds_run_context)

        用于定义在每个数据step结束后执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: end(run_context)

        内部方法，当网络训练结束时释放等待。

        **参数：**

        **run_context**：网络训练运行信息。

    .. py:method:: epoch_begin(run_context)

        用于定义在每个训练epoch开始前执行的回调方法。

        **参数：**

        - **run_context** (RunContext) - 网络训练运行信息。

    .. py:method:: epoch_end(run_context)

        内部方法，不能被调用或重写。通过重写mindspore.train.callback.Callback.epoch_end来释放ds_epoch_begin的等待。

        **参数：**

        **run_context**：网络训练运行信息。

    .. py:method:: step_begin(run_context)

        用于定义在每个训练step开始前执行的回调方法。

        **参数：**

        - **run_context** (RunContext) - 网络训练运行信息。

    .. py:method:: step_end(run_context)

        内部方法，不能被调用或重写。通过重写mindspore.train.callback.Callback.step_end来释放 `ds_step_begin` 的等待。

        **参数：**

        **run_context**：网络训练运行信息。

    .. py:method:: sync_epoch_begin(train_run_context, ds_run_context)

        用于定义在数据epoch开始前，训练epoch结束后执行的回调方法。

        **参数：**

        - **train_run_context**：包含前一个epoch的反馈信息的网络训练运行信息。
        - **ds_run_context**：数据处理管道运行信息。

    .. py:method:: sync_step_begin(train_run_context, ds_run_context)

        用于定义在数据step开始前，训练step结束后执行的回调方法。

        **参数：**

        - **train_run_context**：包含前一个step的反馈信息的网络训练运行信息。
        - **ds_run_context**：数据处理管道运行信息。
