mindspore.dataset.WaitedDSCallback
==================================

.. py:class:: mindspore.dataset.WaitedDSCallback(step_size=1)

    用于自定义与训练回调同步的数据集回调类的抽象基类。

    此类可用于自定义在step或epoch结束后执行的回调方法。
    例如在自动数据增强中根据上一个epoch的loss值来更新增强算子参数配置。

    **参数：**

    - **step_size** (int, optional) - 每个step包含的数据行数。step大小通常与batch大小相等（默认值为1）。

    **样例：**

    >>> from mindspore.dataset import WaitedDSCallback
    >>>
    >>> my_cb = WaitedDSCallback(32)
    >>> # dataset为任意数据集实例
    >>> data = data.map(operations=AugOp(), callbacks=my_cb)
    >>> data = data.batch(32)
    >>> # 定义网络
    >>> model.train(epochs, data, callbacks=[my_cb])

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

        用于定义在每个数据epoch开始前，训练epoch结束后执行的回调方法。

        **参数：**

        - **train_run_context**：包含前一个epoch的反馈信息的网络训练运行信息。
        - **ds_run_context**：数据处理管道运行信息。

    .. py:method:: sync_step_begin(train_run_context, ds_run_context)

        用于定义在每个数据step开始前，训练step结束后执行的回调方法。

        **参数：**

        - **train_run_context**：包含前一个step的反馈信息的网络训练运行信息。
        - **ds_run_context**：数据处理管道运行信息。
