mindspore.ops.UnravelIndex
===========================

.. py:class:: mindspore.ops.UnravelIndex

    将由扁平化索引组成的数组转换为包含坐标数组的元组。

    输入：
        - **indices** (Tensor) - 输入Tensor，其元素将转换为shape为 `dims` 的坐标数组。维度为零维或者一维，其数据类型为int32或int64。
        - **dims** (Tensor) - `indices` 转换之后的shape，输入Tensor。其维度必须为1，数据类型与 `indices` 一致。

    输出：
        - **y** - 输出类型与 `indices` 一致。 `y` 的维度必须是二维或者一维（如果 `indices` 是零维）。

    异常：
        - **TypeError** - 如果 `indices` 和 `dims` 的数据类型不一致。
        - **TypeError** - 如果 `indices` 和 `dims` 的数据类型不是int32或int64。
        - **ValueError** - 如果 `dims` 维度不等于1或者 `indices` 维度不是0或1。
        - **ValueError** - 如果 `indices` 包含负数。
