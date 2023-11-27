mindspore.ops.TensorDump
========================

.. py:class:: mindspore.ops.TensorDump

    将Tensor保存为numpy格式的npy文件。

    文件名会按照执行顺序自动添加前缀。例如，如果 `file` 为 `a`，第一次保存的文件名为 `0_a.npy`，第二次为 `1_a.npy`。

    .. warning::
        - 如果在短时间内保存大量数据，可能会导致设备端内存溢出。可以考虑对数据进行切片，以减小数据规模。
        - 由于数据保存是异步处理的，当数据量过大或主进程退出过快时，可能出现数据丢失的问题，需要主动控制主进程销毁时间，例如使用sleep。

    输入：
        - **file** (str) - 要保存的文件路径。
        - **input_x** (Tensor) - 任意维度的Tensor。

    异常：
        - **TypeError** - 如果 `file` 不是str。
        - **TypeError** - 如果 `input_x` 不是Tensor。
