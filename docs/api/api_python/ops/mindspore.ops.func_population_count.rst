mindspore.ops.population_count
==============================

.. py:function:: mindspore.ops.population_count(input)

    计算二进制数中1的个数。

    **参数：**

    - **input** (Tensor) - 任意维度的Tensor。Ascend平台支持的数据类型为int16、uint16，CPU平台支持的数据类型为int8、int16、int32、int64、uint8、uint16、uint32、uint64。

    **返回：**

    Tensor，shape与 `input` 相同，数据类型为uint8。

    **异常：**

    - **TypeError** - `input` 不是Tensor。
    - **TypeError** - `input` 的数据类型不是int16或uint16（Ascend平台）。
                      `input` 的数据类型不是int8、int16、int32、int64、uint8、uint16、uint32、uint64（CPU平台）。
