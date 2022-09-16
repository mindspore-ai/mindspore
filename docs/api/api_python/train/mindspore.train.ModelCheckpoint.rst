mindspore.train.ModelCheckpoint
===============================

.. py:class:: mindspore.train.ModelCheckpoint(prefix='CKP', directory=None, config=None)

    checkpoint的回调函数。

    在训练过程中调用该方法可以保存网络参数。

    .. note::
        在分布式训练场景下，请为每个训练进程指定不同的目录来保存checkpoint文件。否则，可能会训练失败。
        如何在 `model` 方法中使用此回调函数，默认将会把优化器中的参数保存到checkpoint文件中。

    参数：
        - **prefix** (str) - checkpoint文件的前缀名称。默认值：'CKP'。
        - **directory** (str) - 保存checkpoint文件的文件夹路径。默认情况下，文件保存在当前目录下。默认值：None。
        - **config** (CheckpointConfig) - checkpoint策略配置。默认值：None。

    异常：
        - **ValueError** - 如果prefix参数不是str类型或包含'/'字符。
        - **ValueError** - 如果directory参数不是str类型。
        - **TypeError** - config不是CheckpointConfig类型。

    .. py:method:: end(run_context)

        在训练结束后，会保存最后一个step的checkpoint。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: latest_ckpt_file_name
        :property:

        返回最新的checkpoint路径和文件名。

    .. py:method:: step_end(run_context)

        在step结束时保存checkpoint。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。
