mindspore.dataset.Dataset.sync_update
=====================================

.. py:method:: mindspore.dataset.Dataset.sync_update(condition_name, num_batch=None, data=None)

    释放阻塞条件并使用给定数据触发回调函数。

    参数：
        - **condition_name** (str) - 用于触发发送下一个数据行的条件名称。
        - **num_batch** (Union[int, None]) - 释放的batch（row）数。当 `num_batch` 为None时，将默认为 `sync_wait`  操作指定的值，默认值：None。
        - **data** (Any) - 用户自定义传递给回调函数的数据，默认值：None。
