mindspore.dataset.DSCallback
=============================

.. py:class:: mindspore.dataset.DSCallback(step_size=1)

    数据处理回调类的抽象基类，用户可以基于此类实现自己的回调操作。

    用户可通过 `ds_run_context` 获取数据处理管道相关信息，包括 `cur_epoch_num` (当前epoch数)、 `cur_step_num_in_epoch` (当前epoch的step数)、 `cur_step_num` (当前step数)。

    参数：
        - **step_size** (int, 可选) - 定义相邻的 `ds_step_begin`/`ds_step_end` 调用之间相隔的step数。默认值： ``1`` ，表示每个step都会调用。

    .. py:method:: ds_begin(ds_run_context)

        用于定义在数据处理管道启动前执行的回调方法。

        参数：
            - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_epoch_begin(ds_run_context)

        用于定义在每个数据epoch开始前执行的回调方法。

        参数：
            - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_epoch_end(ds_run_context)

        用于定义在每个数据epoch结束后执行的回调方法。

        参数：
            - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_step_begin(ds_run_context)

        用于定义在指定数据step开始前执行的回调方法。

        参数：
            - **ds_run_context** (RunContext) - 数据处理管道运行信息。

    .. py:method:: ds_step_end(ds_run_context)

        用于定义在指定数据step结束后执行的回调方法。

        参数：
            - **ds_run_context** (RunContext) - 数据处理管道运行信息。
