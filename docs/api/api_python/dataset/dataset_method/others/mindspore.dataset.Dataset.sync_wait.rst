mindspore.dataset.Dataset.sync_wait
===================================

.. py:method:: mindspore.dataset.Dataset.sync_wait(condition_name, num_batch=1, callback=None)

    为同步操作在数据集对象上添加阻塞条件。

    参数：
        - **condition_name** (str) - 用于触发发送下一行数据的条件名称。
        - **num_batch** (int) - 每个epoch开始时无阻塞的batch数。默认值：1。
        - **callback** (function) - `sync_update` 操作中将调用的回调函数。默认值：None。

    返回：
        SyncWaitDataset，添加了阻塞条件的数据集对象。

    异常：
        - **RuntimeError** - 条件名称已存在。
