mindspore.train.LearningRateScheduler
=====================================

.. py:class:: mindspore.train.LearningRateScheduler(learning_rate_function)

    用于在训练期间更改学习率。

    参数：
        - **learning_rate_function** (Function) - 在训练期间更改学习率的函数。

    .. py:method:: step_end(run_context)

        在step结束时更改学习率。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。
