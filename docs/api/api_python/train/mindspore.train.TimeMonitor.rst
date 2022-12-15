mindspore.train.TimeMonitor
===========================

.. py:class:: mindspore.train.TimeMonitor(data_size=None)

    监控训练或推理的时间。

    参数：
        - **data_size** (int) - 表示每隔多少个step打印一次信息。如果程序在训练期间获取到Model的 `batch_num` ，则将把 `data_size` 设为 `batch_num` ，否则将使用 `data_size` 。默认值：None。

    异常：
        - **ValueError** - `data_size` 不是正整数。

    样例：

    .. note::
        运行以下样例之前，需自定义网络LeNet5和数据集准备函数create_dataset。详见 `网络构建 <https://www.mindspore.cn/tutorials/zh-CN/r2.0.0-alpha/beginner/model.html>`_ 和 `数据集 Dataset <https://www.mindspore.cn/tutorials/zh-CN/r2.0.0-alpha/beginner/dataset.html>`_ 。

    .. py:method:: epoch_begin(run_context)

        在epoch开始时记录时间。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: epoch_end(run_context)

        在epoch结束时打印epoch的耗时。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。
