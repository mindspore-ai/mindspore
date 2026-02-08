mindspore.profiler.tensorboard_trace_handler
==============================================

.. py:function:: mindspore.profiler.tensorboard_trace_handler(dir_name=None, worker_name=None, analyse_flag=True, async_mode=False)

    对动态图模式的每一个step，调用该方法进行在线解析。

    参数：
        - **dir_name** (str, 可选) - 指定保存分析结果的目录路径。默认 ``None`` ，表示使用默认的保存路径，默认路径为：``"./data"`` 。
        - **worker_name** (str, 可选) - 指定工程线程名称。默认 ``None`` ，表示使用默认的工程线程名，默认工程线程名为：``"当前操作系统名+进程号"`` 。
        - **analyse_flag** (bool, 可选) - 是否使用在线分析。默认 ``True`` ，表示使用在线分析。
        - **async_mode** (bool, 可选) - 是否使用异步解析模式。默认 ``False`` ，表示使用同步解析模式。
