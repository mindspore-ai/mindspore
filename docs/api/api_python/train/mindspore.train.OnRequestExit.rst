mindspore.train.OnRequestExit
=============================

.. py:class:: mindspore.train.OnRequestExit(save_ckpt=True, save_mindir=True, file_name='Net', directory='./', sig=signal.SIGTERM)

    响应用户关闭请求，退出训练或推理进程，保存checkpoint和mindir。

    在训练开始前，注册OnRequestExit回调，当用户想要退出训练进程并保存训练数据时，可通过发送注册的退出信号"sig"到训练进程。
    训练进程执行完当前step后，保存当前训练状态，包括checkpoint和mindir，然后退出训练过程。

    参数：
        - **save_ckpt** (bool) - 退出训练或推理进程时，是否保存checkpoint。默认值：True。
        - **save_mindir** (bool) - 退出训练或推理进程时，是否保存mindir。默认值：True。
        - **file_name** (str) - 退出训练或推理进程时，保存的checkpoint和mindir的名字，checkpoint文件加.ckpt后缀，mindir文件加.mindir后缀。默认值：'Net'。
        - **directory** (str) - 退出训练或推理进程时，保存的checkpoint和mindir的目录。默认值：'./'。
        - **sig** (int) - 用户注册的退出信号，该信号必须是可捕获可忽略的。当进程收到该信号时，退出训练或者推理。默认值：signal.SIGTERM。

    异常：
        - **ValueError** - `save_ckpt` 不是bool值 。
        - **ValueError** - `save_mindir` 不是字符串。
        - **ValueError** - `file_name` 不是字符串。
        - **ValueError** - `directory` 不是字符串。
        - **ValueError** - `sig` 不是int值，或者是signal.SIGKILL。

    .. py:method:: on_train_begin(run_context)

        在训练开始时，注册用户传入退出信号的处理程序。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_step_end(run_context)

        在训练step结束时，如果接收到退出信号，将 `run_context` 的 `_stop_requested` 属性置为True。在本轮训练结束后，退出训练。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_epoch_end(run_context)

        在训练epoch结束时，如果接收到退出信号，将 `run_context` 的 `_stop_requested` 属性置为True。在本轮训练结束后，退出训练。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_end(run_context)

        在训练结束时，如果接收到退出信号，根据用户配置，保存checkpoint和mindir。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_begin(run_context)

        在推理开始时，注册用户传入退出信号的处理程序。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_step_end(run_context)

        在推理step结束时，如果接收到退出信号，将 `run_context` 的 `_stop_requested` 属性置为True。在本轮推理结束后，退出推理。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_eval_end(run_context)

        在推理结束时，如果接收到退出信号，根据用户配置，保存checkpoint和mindir。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。
