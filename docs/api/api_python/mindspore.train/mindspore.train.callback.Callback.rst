.. py:class:: mindspore.train.callback.Callback

    用于构建Callback函数的基类。Callback函数是一个上下文管理器，在运行模型时被调用。
    可以使用此机制进行一些自定义操作。

    Callback函数可以在step或epoch开始前或结束后执行一些操作。
    要创建自定义Callback，需要继承Callback基类并重载它相应的方法，有关自定义Callback的详细信息，请查看
    `Callback <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html>`_。

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
