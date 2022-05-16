mindspore.Callback
===================

.. py:class:: mindspore.Callback

    用于构建Callback函数的基类。Callback函数是一个上下文管理器，在运行模型时被调用。
    可以使用此机制进行一些自定义操作。

    Callback类的每个方法对应了训练或推理过程的不同阶段，这些方法有相同的入参 `run_context`，用于保存模型
    训练或推理过程模型的相关信息。定义Callback子类或自定义Callback时，请根据需要重写对应的方法。

    自定义Callback场景下，在类方法中通过 `RunContext.original_args()` 方法可以获取模型训练或推理过程中已有
    的上下文信息，此信息为一个存储了已有属性的字典型变量；用户也可以在此信息中添加其他的自定义属性；此外，
    通过调用 `request_stop` 方法来停止训练过程。有关自定义Callback的具体用法，请查看
    `Callback <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debug.html>`_。

    .. py:method:: begin(run_context)

        在网络执行之前被调用一次。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: end(run_context)

        网络执行后被调用一次。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: epoch_begin(run_context)

        在每个epoch开始之前被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: epoch_end(run_context)

        在每个epoch结束后被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: step_begin(run_context)

        在每个step开始之前被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: step_end(run_context)

        在每个step完成后被调用。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。
