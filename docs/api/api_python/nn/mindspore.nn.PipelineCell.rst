mindspore.nn.PipelineCell
=========================

.. py:class:: mindspore.nn.PipelineCell(network, micro_size)

    将MiniBatch切分成更细粒度的MicroBatch，用于流水线并行的训练中。

    .. note::
        micro_size必须大于或等于流水线stage的个数。

    参数：
        - **network** (Cell) - 要修饰的目标网络。
        - **micro_size** (int) - MicroBatch大小。
