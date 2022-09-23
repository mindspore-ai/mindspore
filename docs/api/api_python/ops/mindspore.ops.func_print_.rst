mindspore.ops.print\_
=====================

.. py:function:: mindspore.ops.print_(*input_x)

    将输入数据进行打印输出。

    默认打印在屏幕上。也可以保存在文件中，通过 `context` 设置 `print_file_path` 参数。一旦设置，输出将保存在指定文件中。通过函数 :func:`mindspore.parse_print` 可以重新加载数据。获取更多信息，请查看 :func:`mindspore.set_context` 和 :func:`mindspore.parse_print` 。

    .. note::
        在PyNative模式下，请使用Python print函数。在Ascend平台上的Graph模式下，bool、int和float将被转换为Tensor进行打印，str保持不变。
        该方法用于代码调试。当同时print大量数据时，为了保证主进程不受影响，可能会丢失一些数据。如果需要记录完整数据，推荐使用 `Summary` 功能，具体可查看
        `Summary <https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.9/summary_record.html?highlight=summary#>`_ 。

    参数：
        - **input_x** (Union[Tensor, bool, int, float, str]) - print_的输入。支持多个输入，用'，'分隔。

    返回：
        无效返回值，应忽略。

    异常：
        - **TypeError** - `input_x` 不是Tensor、bool、int、float或str。
