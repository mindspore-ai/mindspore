mindspore.async_ckpt_thread_status
=======================================

.. py:class:: mindspore.async_ckpt_thread_status()

    获取异步保存checkpoint文件线程的状态。

    在执行异步保存checkpoint时，可以通过该函数获取线程状态以确保写入checkpoint文件已完成。

    **返回：**

    True，异步保存checkpoint线程正在运行。
    False，异步保存checkpoint线程未运行。
