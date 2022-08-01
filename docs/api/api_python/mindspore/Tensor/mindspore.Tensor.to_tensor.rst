mindspore.Tensor.to_tensor
==========================

.. py:method:: mindspore.Tensor.to_tensor(slice_index=None, shape=None, opt_shard_group=None)

    返回init_data()的结果，并获取此Tensor的数据。

    .. note::
        不建议使用 `to_tensor` 。请使用 `init_data` 。

    参数：
        - **slice_index** (int) - 参数切片的索引。在初始化参数切片的时候使用，保证使用相同切片的设备可以生成相同的Tensor。默认值：None。
        - **shape** (list[int]) - 切片的shape，在初始化参数切片时使用。默认值：None。
        - **opt_shard_group** (str) - 优化器分片组，在自动或半自动并行模式下用于获取参数切片的分片。默认值：None。

    返回：
        Tensor，shape和数据类型与原Tensor相同。

    异常：
        - **TypeError** - `indices` 的数据类型既不是int32，也不是int64。
        - **ValueError** - Tensor的shape长度小于 `indices` 的shape的最后一个维度。