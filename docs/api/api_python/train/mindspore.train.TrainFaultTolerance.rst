mindspore.train.TrainFaultTolerance
===================================

.. py:class:: mindspore.train.TrainFaultTolerance(ckpt_save_path=None, **kwargs)

    该回调函数用于开启 `MindIO的TTP特性 <https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/ref/mindiottp/mindiotft001.html>`_，会嵌入训练的流程，完成TTP的初始化、上报、异常处理等操作。

    .. note::
        该特性仅支持Ascend后端的静态图模式，并且只支持sink_size值小于等于1的场景。

    参数：
        - **ckpt_save_path** (str，可选) - 异常发生时ckpt保存的路径，该路径是一个目录。保存时，会在该目录下创建新的名为‘ttp_saved_checkpoints-step_{cur_step_num}’目录。默认值为: ``None``。
        - **kwargs** (dict) - 其他字典类型参数。当参数 `ckpt_save_path` 的值为 ``None`` 时， `kwargs` 必须包含一个名为 `ckpt_save_fn` 的参数，该参数指向一个保存Checkpoint的函数。 `ckpt_save_fn` 的函数原型为 ``def save_ckpt(cb_params, append_dict)`` 。当同时提供 `ckpt_save_path` 和 `ckpt_save_fn` 参数，则优先使用 `ckpt_save_fn` 参数。

    异常：
        - **Exception** - TTP初始化失败，会抛出Exception异常。
        - **ModuleNotFoundError** - Mindio TTP whl包未安装。

    样例：

    .. note::
        在运行TrainFaultTolerance的用例之前，需要配置相应的环境变量。推荐使用msrun进行分布式的启动，参考 `msrun启动方式 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html>`_。用例应该在4张卡上运行。

    .. py:method:: end(run_context)

        训练结束，解注册MindIO TTP。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: get_optimizer_wrapper(origin_opt_cls)
        :classmethod:

        使用TFT功能时的优化器类封装函数。

        参数：
            - **origin_opt_cls** (Class) - 原优化器类。

    .. py:method:: on_train_begin(run_context)

        训练开始时，向MindIO TTP注册训练时参数。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_step_end(run_context)

        每个step完成时，进行MindIO TTP的上报。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。
