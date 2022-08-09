mindspore.ops.population_count
==============================

.. py:function:: mindspore.ops.population_count(input_x)

    计算二进制数中1的个数。

    **参数：**

    - **input_x** (Tensor) - 任意维度的Tensor。Ascend平台支持的数据类型为int16、uint16，CPU和GPU平台支持的数据类型为int8、int16、int32、int64、uint8、uint16、uint32、uint64。

    **返回：**

    Tensor，shape与 `input_x` 相同，数据类型为uint8。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。
    - **TypeError** - `input_x` 的数据类型不是int16或uint16（Ascend平台）。
    - **TypeError** - `input_x` 的数据类型不是int8、int16、int32、int64、uint8、uint16、uint32、uint64（CPU和GPU平台）。
