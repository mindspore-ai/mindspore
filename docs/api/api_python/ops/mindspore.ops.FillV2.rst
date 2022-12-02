mindspore.ops.FillV2
====================

.. py:class:: mindspore.ops.FillV2

    创建一个Tensor，其shape由 `shape` 指定，其值则由 `value` 进行填充。

    输入：
        - **shape** (Tensor) - 1-D Tensor，指定了输出Tensor的shape。
          其数据类型必须是int32或int64。
        - **value** (Tensor) - 一个标量Tensor，其值用于填充输出Tensor。
          `value` 必须是0-D的，且其数据类型必须是以下之一：
          bool、int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64。

    输出：
        - **y** (Tensor) - Tensor，其shape和值如上所述。

    异常：
        - **ValueError** - 如果 `shape` 不是1-D Tensor。
        - **TypeError** - 如果 `shape` 的数据类型不是int32或者int64。
        - **ValueError** - 如果 `value` 不是0-D Tensor。
        - **ValueError** - 如果输出元素的数量多于1000000。
