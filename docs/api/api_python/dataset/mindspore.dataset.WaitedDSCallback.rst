mindspore.dataset.WaitedDSCallback
==================================

.. py:class:: mindspore.dataset.WaitedDSCallback(step_size=1)

    阻塞式数据处理回调类的抽象基类，用于与训练回调类 `mindspore.train.Callback <https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Callback.html#mindspore.train.Callback>`_ 的同步。

    可用于在step或epoch开始前执行自定义的回调方法，例如在自动数据增强中根据上一个epoch的loss值来更新增强操作参数配置。

    用户可通过 `train_run_context` 获取网络训练相关信息，如 `network` 、 `train_network` 、 `epoch_num` 、 `batch_num` 、 `loss_fn` 、 `optimizer` 、 `parallel_mode` 、 `device_number` 、 `list_callback` 、 `cur_epoch_num` 、 `cur_step_num` 、 `dataset_sink_mode` 、 `net_outputs` 等，详见 `mindspore.train.Callback <https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Callback.html#mindspore.train.Callback>`_ 。

    用户可通过 `ds_run_context` 获取数据处理管道相关信息，包括 `cur_epoch_num` (当前epoch数)、 `cur_step_num_in_epoch` (当前epoch的step数)、 `cur_step_num` (当前step数)。

    .. note:: 注意，第2个step或epoch开始时才会触发该调用。

    参数：
        - **step_size** (int, 可选) - 每个step包含的数据行数。通常step_size与batch_size一致，默认值：1。

    .. py:method:: sync_epoch_begin(train_run_context, ds_run_context)

        用于定义在数据epoch开始前，训练epoch结束后执行的回调方法。

        参数：
            - **train_run_context** - 包含前一个epoch的反馈信息的网络训练运行信息。
            - **ds_run_context** - 数据处理管道运行信息。

    .. py:method:: sync_step_begin(train_run_context, ds_run_context)

        用于定义在数据step开始前，训练step结束后执行的回调方法。

        参数：
            - **train_run_context** - 包含前一个step的反馈信息的网络训练运行信息。
            - **ds_run_context** - 数据处理管道运行信息。
