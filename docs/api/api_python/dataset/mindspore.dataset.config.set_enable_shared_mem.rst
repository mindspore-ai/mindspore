mindspore.dataset.config.set_enable_shared_mem
===============================================

.. py:function:: mindspore.dataset.config.set_enable_shared_mem(enable)

    设置共享内存标志的是否启用。如果 `shared_mem_enable` 为True，则使用共享内存队列将数据传递给为数据集操作而创建的进程，而这些数据集操作将设置 `python_multiprocessing` 为True。

    .. note::
        Windows和MacOS平台尚不支持 `set_enable_shared_mem` 。

    参数：
        - **enable** (bool) - 表示当 `python_multiprocessing` 为True时，是否在数据集操作中使用共享内存。

    异常：
        - **TypeError** - `enable` 不是bool类型。
