mindspore.ops.PopulationCount
=============================

.. py:class:: mindspore.ops.PopulationCount

    计算二进制数中1的个数。

    更多参考详见 :func:`mindspore.ops.population_count`。

    输入：
        - **input_x** (Tensor) - 任意维度的Tensor。Ascend平台支持的数据类型为int16、uint16，CPU和GPU平台支持的数据类型为int8、int16、int32、int64、uint8、uint16、uint32、uint64。

    输出：
        Tensor，shape与 `input_x` 相同，数据类型为uint8。
