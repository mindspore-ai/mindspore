mindspore.train.FlopsUtilizationCollector
=========================================

.. py:class:: mindspore.train.FlopsUtilizationCollector(data_size, computility=1, full_flops=True, enable_ma_collector=False)

    FlopsUtilizationCollector接口统计模型利用率信息MFU，硬件利用率信息HFU。

    当前接口只统计MatMul、BatchMatMul、flash_attention_score、Conv2D算子的正反向flops信息。

    只支持静态图静态shape模式。

    参数：
        - **data_size** (int) - 表示每隔多少个step打印一次信息。

        - **computility** (int，可选) - 表示每张计算卡的峰值算力。默认值： ``1`` 。

        - **full_flops** (bool，可选) - 表示是否统计完整的模型信息。如果设置为True，会统计完整的模型信息；如果设置为False，将会统计对应每张卡的分片模型信息。默认值： ``True`` 。

        - **enable_ma_collector** (bool，可选) - 表示是否将flops写日志，提供给云上任务进行采集。默认值： ``False`` 。

    异常：
        - **TypeError** - `data_size` 不是正整数。
        - **TypeError** - `full_flops` 不是布尔类型。
        - **TypeError** - `enable_ma_collector` 不是布尔类型。
        - **AssertionError** - 训练模式不是静态图或者不是静态shape。

    .. py:method:: step_begin(run_context)

        在step开始时记录时间。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: step_end(run_context)

        在step结束时打印模型利用率信息MFU，硬件利用率信息HFU。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。
