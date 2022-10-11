mindspore.ops.population_count
==============================

.. py:function:: mindspore.ops.population_count(input_x)

    逐元素计算population count（又称bitsum, bitcount）。
    对于 `input_x` 中的每个entry，计算该entry的二进制表示中的1比特的数量。

    参数：
        - **input_x** (Tensor) - 任意维度的Tensor。Ascend平台支持的数据类型为int16、uint16，CPU和GPU平台支持的数据类型为int8、int16、int32、int64、uint8、uint16、uint32、uint64。

    返回：
        Tensor，shape与 `input_x` 相同，数据类型为uint8。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `input_x` 的数据类型不是int16或uint16（Ascend平台）。
        - **TypeError** - `input_x` 的数据类型不是int8、int16、int32、int64、uint8、uint16、uint32、uint64（CPU和GPU平台）。
