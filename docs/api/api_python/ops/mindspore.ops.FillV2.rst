mindspore.ops.FillV2
====================

.. py:class:: mindspore.ops.FillV2

    创建一个Tensor，其shape由 `shape` 指定，其值则由 `value` 进行填充。

    输入：
        - **shape** (Union[Tuple[int], Tensor[int]]) - 1-D Tensor或Tuple，指定了输出Tensor的shape。
          其数据类型必须是int32或int64。
        - **value** (Tensor) - `value` 是零维Tensor，其值用于填充输出 `y` 。

    输出：
        - **y** (Tensor) - Tensor，其shape和值如上所述。

    异常：
        - **TypeError** - 如果 `shape` 不是1-D Tensor或Tuple。
        - **TypeError** - 如果 `shape` 的数据类型不是int32或者int64。
        - **ValueError** - 如果 `value` 不是零维Tensor。
