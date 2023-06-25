mindspore.ops.IndexPut
=======================

.. py:class:: mindspore.ops.IndexPut(accumulate=0)

    根据 `indices` 中的下标值，使用 `x2` 中的数值替换 `x1` 中的相应元素的值。

    参数：
        - **accumulate** (int) - 如果 `accumulate` 被设置为1 ， `x2` 中的元素被累加到 `x1` 的相应元素上；
          如果为 ``0``，用 `x2` 中的元素取代 `x2` 的相应元素。默认值： ``0`` 。

    输入：
        - **x1** (Tensor) - 被执行替换操作的Tensor, 维度大于等于1。
        - **x2** (Tensor) - 数据类型和 `x1` 一致的一维的Tensor。如果其size为1，则shape将被广播为 `x1` 的shape。
        - **indices** (tuple[Tensor], list[Tensor]) - 元素类型是int32或者int64, 用于对 `x1` 中的元素进行索引。
          `indices` 中的tensor的秩应为1-D， `indices` 中tensor的size应小于 `x1` 的秩，indices中的tensor应是可广播的。

    输出：
        Tensor, 其数据类型和shape与  `x1` 相同。

    异常：
        - **TypeError** - 如果 `x1` 的dtype与 `x2` 的dtype不同。
        - **TypeError** - 如果 `indices` 不是tuple[Tensor]或者list[Tensor]。
        - **TypeError** - 如果 `indices` 中的tensor的dtype不是int32或者int64。
        - **TypeError** - 如果 `indices` 中的tensor的dtype是不一致的。
        - **TypeError** - 如果 `accumulate` 的dtype不是int。
        - **ValueError** - 如果 `x2` 的秩不是1-D。
        - **ValueError** - 当 `x1` 的rank与 `indices` 的size相等时，如果 `x2` 的size不为1
          或者不为 `indices` 中Tensor的最大size。
        - **ValueError** - 当 `x1` 的rank大于 `indices` 的size时，如果 `x2` 的size不为1
          或者不为 `x1` 的最后一维的shape。
        - **ValueError** - 如果 `indices` 中的tensor的秩不是1-D。
        - **ValueError** - 如果 `indices` 中的tensor不是可广播的。
        - **ValueError** - 如果 `indices` 的size大于 `x1` 的秩。
        - **ValueError** - 如果 `accumulate` 的值不是0或1。
