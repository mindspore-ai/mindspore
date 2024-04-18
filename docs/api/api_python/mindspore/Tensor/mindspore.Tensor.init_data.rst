mindspore.Tensor.init_data
==========================

.. py:method:: mindspore.Tensor.init_data(slice_index=None, shape=None, opt_shard_group=None)

    获取此Tensor的数据。

    .. note:: 对于同一个Tensor，只可以调用一次 `init_data` 函数。

    参数：
        - **slice_index** (int) - 参数切片的索引。在初始化参数切片的时候使用，保证使用相同切片的设备可以生成相同的Tensor。默认值： ``None`` 。
        - **shape** (list[int]) - 切片的shape，在初始化参数切片时使用。默认值： ``None`` 。
        - **opt_shard_group** (str) - 优化器分片组，在自动或半自动并行模式下用于获取参数的切片。关于优化器分组，请参考 `优化器并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/optimizer_parallel.html>`_ 。默认值： ``None`` 。

    返回：
        初始化的Tensor。