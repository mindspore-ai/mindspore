mindspore.train.MindIOTTPAdapter
================================

.. py:class:: mindspore.train.MindIOTTPAdapter(controller_ip, controller_port, ckpt_save_path)

    该回调用于开启 `MindIO的TTP特性 <https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/mindio/mindiottp/mindiottp001.html>`_，
    该CallBack会嵌入训练的流程，完成TTP 的初始化、上报、异常处理等操作。

    .. note::
        该特性仅支持Ascend GE LazyInline 模式，且满足 pipeline 流水并行大于1的要求。

    参数：
        - **controller_ip** (str) - TTP controller 的IP地址, 该参数用于启动TTP的controller。
        - **controller_port** (int) - TTP controller 的IP端口, 该参数用于启动TTP的controller和processor。
        - **ckpt_save_path** (str) -  异常发生时ckpt保存的路径，该路径是一个目录，ckpt的异常保存时会在该录下创建新的名为‘ttp_saved_checkpoints-{cur_epoch_num}_{cur_step_num}’目录。

    异常：
        - **Exception** - TTP 初始化失败，会对外抛Exception异常。
        - **ModuleNotFoundError** - Mindio TTP whl 包未安装。

    .. py:method:: load_checkpoint_with_backup(ckpt_file_path, strategy_file_path, net)

        加载指定的checkpoint文件到网络中，如果配置的checkpoint文件没有，基于strategy文件获取备份的checkpoint进行加载。

        .. note::
            该接口必须在通信初始化后调用，因为内部需要获取集群的信息。

        参数：
            - **ckpt_file_path** (str) - 需要加载的checkpoint文件。
            - **strategy_file_path** (str) - 当前卡的strategy 文件。
            - **net** (Cell) - 需要加载权重的网络。

        异常：
            - **ValueError** - 加载checkpoint文件失败。

        返回：
            Dict:  加载后的checkpoint权重。

    .. py:method:: on_train_step_end(run_context)

         在第一次step完成进行MindIO TTP的初始化， 每个step完成时进行MindIO TTP的上报。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: wrapper_ttp_persist(func)

        对出传入的函数进行TTP异常处理的封装。

        参数：
            - **func** (function) - 需要封装的训练函数

        返回：
            Function: 如果TTP使能，则返回封装后的函数，否则返回原函数。
