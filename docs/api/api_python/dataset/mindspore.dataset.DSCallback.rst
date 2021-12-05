mindspore.dataset.DSCallback
=============================

.. py:class:: mindspore.dataset.DSCallback(step_size=1)

    用于自定义数据回调类的抽象基类。

    **参数：**

    - **step_size** (int, optional) - 调用 `ds_step_begin` 和 `ds_step_end` 之间间隔的step数（默认为1）。

    **样例：**

    >>> from mindspore.dataset import DSCallback
    >>>
    >>> class PrintInfo(DSCallback):
    ...     def ds_epoch_end(self, ds_run_context):
    ...         print(cb_params.cur_epoch_num)
    ...         print(cb_params.cur_step_num)
    >>>
    >>> # dataset为任意数据集实例，op为任意数据处理算子
    >>> dataset = dataset.map(operations=op, callbacks=PrintInfo())

    .. py:method:: ds_begin(ds_run_context)

        用于定义在数据处理管道启动前执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_epoch_begin(ds_run_context)

        用于定义在每个数据epoch开始前执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_epoch_end(ds_run_context)

        用于定义在每个数据epoch结束后执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_step_begin(ds_run_context)

        用于定义在每个数据step开始前执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_step_end(ds_run_context)

         用于定义在每个数据step结束后执行的回调方法。

        **参数：**

        - **ds_run_context** (RunContext) - 数据处理管道运行信息。
