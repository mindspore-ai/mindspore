.. py:class:: mindspore.train.callback.History(has_trained_epoch=0)

    将网络输出的相关信息记录到 `History` 对象中。

    用户不自定义训练网络或评估网络情况下，记录的内容将为损失值；用户自定义了训练网络/评估网络的情况下，如果定义的网络返回 `Tensor` 或 `numpy.ndarray`，则记录此返回值均值，如果返回 `tuple` 或 `list`，则记录第一个元素。

    .. note::
        通常使用在 `mindspore.Model.train` 中。

    **参数：**

    - **has_trained_epoch** (int) - 表示已经训练了多少个epoch，如果设置了该参数，History将监控该数值之后epoch的网络输出信息。默认值：0。

    **异常：**

    - **ValueError** - 当 `has_trained_epoch` 不是整数或小于零。

    .. py:method:: begin(run_context)

        训练开始时初始化History对象的epoch属性。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: epoch_end(run_context)

        epoch结束时记录网络输出的相关信息。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。
