mindspore.train.History
=======================

.. py:class:: mindspore.train.History

    将网络输出和评估指标的相关信息记录到 `History` 对象中。

    - 用户不自定义训练网络或评估网络情况下，则记录损失值。
    - 用户自定义了训练网络或评估网络的情况下，

      - 如果定义的网络返回 `Tensor` 或 `numpy.ndarray`，则记录此返回值均值。
      - 如果返回 `tuple` 或 `list`，则记录第一个元素。

    .. note::
        通常使用在 :func:`mindspore.train.Model.train` 和 :func:`mindspore.train.Model.fit` 中。

    .. py:method:: begin(run_context)

        训练开始时，初始化History对象的epoch属性。

        参数：
            - **run_context** (RunContext) - 包含模型的上下文环境。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: epoch_end(run_context)

        epoch结束时，记录网络输出的第一个元素和评估指标的相关信息。

        参数：
            - **run_context** (RunContext) - 包含模型的上下文环境。详情请参考 :class:`mindspore.train.RunContext`。
