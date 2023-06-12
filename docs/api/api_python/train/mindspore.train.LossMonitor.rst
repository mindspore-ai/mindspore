mindspore.train.LossMonitor
===========================

.. py:class:: mindspore.train.LossMonitor(per_print_times=1)

    训练场景下，监控训练的loss；边训练边推理场景下，监控训练的loss和推理的metrics。

    如果loss是NAN或INF，则终止训练。

    .. note::
        如果 `per_print_times` 为0，则不打印loss。

    参数：
        - **per_print_times** (int) - 表示每隔多少个step打印一次loss。默认值：1。

    异常：
        - **ValueError** - 当 `per_print_times` 不是整数或小于零。

    .. py:method:: on_train_epoch_end(run_context)

        LossMoniter用于 `model.fit`，即边训练边推理场景时，打印训练的loss和当前epoch推理的metrics。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: step_end(run_context)

        step结束时打印训练loss。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。
