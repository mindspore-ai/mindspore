mindspore.train.Callback
========================

.. py:class:: mindspore.train.Callback

    用于构建Callback函数的基类。Callback函数是一个上下文管理器，在运行模型时被调用。
    可以使用此机制进行一些自定义操作。

    Callback类的每个方法对应了训练或推理过程的不同阶段，这些方法有相同的入参 `run_context`，用于保存训练或推理过程中模型的相关信息。定义Callback子类或自定义Callback时，请根据需要重写名称前缀为"on_train"或"on_eval"的方法，否则自定义的Callback在 `model.fit` 中使用时会产生错误。

    自定义Callback场景下，在类方法中通过 `RunContext.original_args()` 方法可以获取模型训练或推理过程中已有
    的上下文信息，此信息为一个存储了已有属性的字典型变量。用户也可以在此信息中添加其他的自定义属性。此外，
    通过调用 `request_stop` 方法来停止训练过程。有关自定义Callback的具体用法，请查看
    `Callback <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debug.html>`_。

    .. py:method:: begin(run_context)

        在网络执行之前被调用一次。与 `on_train_begin` 和 `on_eval_begin` 方法具有兼容性。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: end(run_context)

        网络执行后被调用一次。与 `on_train_end` 和 `on_eval_end` 方法具有兼容性。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: epoch_begin(run_context)

        在每个epoch开始之前被调用。与 `on_train_epoch_begin` 和 `on_eval_epoch_begin` 方法具有兼容性。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: epoch_end(run_context)

        在每个epoch结束后被调用。与 `on_train_epoch_end` 和 `on_eval_epoch_end` 方法具有兼容性。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_begin(run_context)

        在网络执行推理之前调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_end(run_context)

        网络执行推理之后调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_epoch_begin(run_context)

        在推理的epoch开始之前被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_epoch_end(run_context)

        在推理的epoch结束后被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_step_begin(run_context)

        在推理的每个step开始之前被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_step_end(run_context)

        在推理的每个step完成后被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_begin(run_context)

        在网络执行训练之前调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_end(run_context)

        网络训练执行结束时调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_epoch_begin(run_context)

        在训练的每个epoch开始之前被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_epoch_end(run_context)

        在训练的每个epoch结束后被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_step_begin(run_context)

        在训练的每个step开始之前被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_step_end(run_context)

        在训练的每个step完成后被调用。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: step_begin(run_context)

        在每个step开始之前被调用。与 `on_train_step_begin` 和 `on_eval_step_begin` 方法具有兼容性。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: step_end(run_context)

        在每个step完成后被调用。与 `on_train_step_end` 和 `on_eval_step_end` 方法具有兼容性。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。
