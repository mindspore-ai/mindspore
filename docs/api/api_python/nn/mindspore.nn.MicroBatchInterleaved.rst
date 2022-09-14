mindspore.nn.MicroBatchInterleaved
==================================

.. py:class:: mindspore.nn.MicroBatchInterleaved(network, interleave_num=2)

    这个函数的作用是将输入在第0维度拆成 `interleave_num` 份，然后执行包裹的cell的计算。
    使用场景：当在半自动模式以及网络中存在模型并行时，第1份的切片数据的前向计算同时，第2份的数据将会进行模型并行的通信，以此来达到通信计算并发的性能加速。

    .. note::
        传入的 `network` 的输出只能是单个Tensor。

    参数：
        - **network** (Cell) - 需要封装的网络。
        - **interleave_num** (int) - batch size的拆分份数，默认值为2。

    输入：
        tuple[Tensor]，与传入的 `network` 的输入一致。

    输出：
        传入的network的输出。
