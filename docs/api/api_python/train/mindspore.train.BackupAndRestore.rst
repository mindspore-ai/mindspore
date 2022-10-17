mindspore.train.BackupAndRestore
================================

.. py:class:: mindspore.train.BackupAndRestore(backup_dir, save_freq="epoch", delete_checkpoint=True)

    在训练过程中备份和恢复训练参数的回调函数。

    .. note::
        只能在训练过程使用这个方法。

    参数：
        - **backup_dir** (str) - 保存和恢复checkpoint文件的路径。
        - **save_freq** (Union['epoch', int]) - 当设置为'epoch'时，在每个epoch进行备份，当设置为整数时，将在每隔 `save_freq` 个epoch进行备份。默认值：'epoch'。
        - **delete_checkpoint** (bool) - 如果 `delete_checkpoint=True` ，将在训练结束的时候删除备份文件，否则保留备份文件。默认值：True。

    异常：
        - **ValueError** - 如果 `backup_dir` 参数不是str类型。
        - **ValueError** - 如果 `save_freq` 参数不是'epoch'或str类型。
        - **ValueError** - 如果 `delete_checkpoint` 参数不是bool类型。

    .. py:method:: on_train_begin(run_context)

        在训练开始时，加载备份的checkpoint文件。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_epoch_end(run_context)

        在每个epoch结束时，判断是否需要备份checkpoint文件。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: on_train_end(run_context)

        在训练结束时，判断是否删除备份的checkpoint文件。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。