mindspore.train.TimeMonitor
===========================

.. py:class:: mindspore.train.TimeMonitor(data_size=None, data_time=False)

    监控训练或推理的时间。

    参数：
        - **data_size** (int，可选) - 表示每隔多少个step打印一次信息。如果程序在训练期间获取到Model的 `batch_num` ，则将 `data_size` 设为 `batch_num` 的值，否则将直接使用 `data_size` 的值。如果程序在训练期间没有获取到Model的 `batch_num` 且没有设置 `data_size`，程序将会报错。默认值： ``None`` 。

        - **data_time** (bool，可选) - 表示是否打印在Host侧获取数据的时间。请注意：在非数据集下沉模式时，数据获取和网络计算是同步执行的，而在数据集下沉模式时，它们是异步执行的。默认值： ``False`` 。

    异常：
        - **ValueError** - `data_size` 不是正整数。
        - **TypeError** - `data_time` 不是布尔类型。

    .. py:method:: epoch_begin(run_context)

        在epoch开始时记录时间。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: epoch_end(run_context)

        在epoch结束时打印epoch的耗时。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_step_begin(run_context)

        在step开始时记录时间。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_step_end(run_context)

        在step结束时记录时间。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。
