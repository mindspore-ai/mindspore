mindspore.async_ckpt_thread_status
=======================================

.. py:function:: mindspore.async_ckpt_thread_status()

    获取异步保存checkpoint文件线程的状态。

    在执行异步保存checkpoint时，判断异步线程是否执行完毕。

    返回：
        Bool: True，异步保存checkpoint线程正在运行。False，异步保存checkpoint线程未运行。