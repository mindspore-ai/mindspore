mindspore.train.ModelCheckpoint
===============================

.. py:class:: mindspore.train.ModelCheckpoint(prefix='CKP', directory=None, config=None)

    checkpoint的回调函数。

    在训练过程中，调用该类可以保存网络参数。

    .. note::
        在分布式训练场景下，请为每个训练进程指定不同的目录，来保存checkpoint文件。否则，可能会训练失败。如果在 `Model <https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Model.html>`_ 方法中使用此回调函数，默认会把优化器中的参数保存到checkpoint文件中。

    参数：
        - **prefix** (Union[str, callable object]，可选) - checkpoint文件的前缀名称，或者用来生成名称的可调用对象。默认值： ``'CKP'`` 。
        - **directory** (Union[str, callable object]，可选) - 保存checkpoint文件的文件夹路径，或者用来生成路径的可调用对象。默认情况下，文件保存在当前目录下。默认值： ``None`` 。
        - **config** (CheckpointConfig，可选) - checkpoint策略配置。默认值： ``None`` 。

    异常：
        - **ValueError** - 如果 `prefix` 参数不是str类型或包含'/'字符，且不是可调用对象。
        - **ValueError** - 如果 `directory` 参数不是str类型，且不是可调用对象。
        - **TypeError** - `config` 不是CheckpointConfig类型。

    .. py:method:: end(run_context)

        在训练结束后，会保存最后一个step的checkpoint。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。

    .. py:method:: latest_ckpt_file_name
        :property:

        返回最新的checkpoint路径和文件名。

    .. py:method:: step_end(run_context)

        在step结束时保存checkpoint。

        参数：
            - **run_context** (RunContext) - 包含模型的一些基本信息。详情请参考 :class:`mindspore.train.RunContext`。
