mindspore.train.OnRequestExit
===========================

.. py:class:: mindspore.train.OnRequestExit(save_ckpt=True, save_mindir=True, file_name='Net', directory='./', sig=signal.SIGUSR1)

    响应用户关闭请求，退出训练或推理进程，保存checkpoint和mindir。

    参数：
        - **save_ckpt** (bool) - 退出训练或推理进程时，是否保存checkpoint。默认值：True。
        - **save_mindir** (bool) - 退出训练或推理进程时，是否保存mindir。默认值：True。
        - **file_name** (str) - 退出训练或推理进程时，保存的checkpoint和mindir的名字，checkpoint文件加.ckpt后缀，mindir文件加.mindir后缀。默认值：'Net'。
        - **directory** (str) - 退出训练或推理进程时，保存的checkpoint和mindir的目录。默认值：'./'。
        - **sig** (int) - 退出训练或推理进程时，保存的checkpoint和mindir的目录。默认值：signal.SIGUSR1。

    异常：
        - **ValueError** - `save_ckpt` 不是bool值 。
        - **ValueError** - `save_mindir` 不是字符串。
        - **ValueError** - `file_name` 不是字符串。
        - **ValueError** - `directory` 不是字符串。
        - **ValueError** - `sig` 不是int数，或者是signal.SIGKILL。

    .. py:method:: on_train_begin(run_context)

        在训练开始时，注册用户传入停止信号的处理程序。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.RunContext`。

    .. py:method:: on_train_step_end(run_context)

        在训练step结束时，根据是否接收到停止信号，将`run_context`的`_stop_requested`属性置为True。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.RunContext`。
        .. py:method:: on_train_step_end(run_context)

    .. py:method:: on_train_epoch_end(run_context)

        在训练epoch结束时，根据是否接收到停止信号，将`run_context`的`_stop_requested`属性置为True。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.RunContext`。
        .. py:method:: on_train_step_end(run_context)

    .. py:method:: on_train_end(run_context)

        在训练结束时，根据是否接收到停止信号，保存checkpoint或者mindir。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.RunContext`。

    .. py:method:: on_eval_begin(run_context)

        在推理开始时，注册用户传入停止信号的处理程序。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.RunContext`。

    .. py:method:: on_eval_step_end(run_context)

        在推理step结束时，根据是否接收到停止信号，将`run_context`的`_stop_requested`属性置为True。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.RunContext`。

    .. py:method:: on_eval_end(run_context)

        在推理结束时，根据是否接收到停止信号，保存checkpoint或者mindir。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.RunContext`。
