mindspore.Tensor.index_put
==========================

.. py:method:: mindspore.Tensor.index_put(indices, values, accumulate=False)

    返回一个Tensor。根据 `indices` 中的下标值，使用 `values` 中的数值替换Tensor本身中的相应元素的值。

    参数：
        - **indices** (tuple[Tensor], list[Tensor]) - 元素类型是int32或者int64, 用于对Tensor本身中的元素进行索引。
          `indices` 中的Tensor的秩应为1-D，`indices` 的size应小于或等于Tensor本身的秩，indices中的Tensor应是可广播的。
        - **values** (Tensor) - 一个一维的Tensor, 其数据类型与Tensor本身相同。如果其size为1，则它是可广播的。
        - **accumulate** (bool) - 如果 `accumulate` 被设置为True， `values` 中的元素被累加到Tensor本身的相应元素上；
          否则，用 `values` 中的元素取代Tensor本身的相应元素。默认值: False。

    返回：
        Tensor, 其数据类型和shape与Tensor本身相同。

    异常：
        - **TypeError** - 如果Tensor本身的dtype与 `values` 的dtype不同。
        - **TypeError** - 如果 `indices` 的dtype不是tuple[Tensor]或者list[Tensor]。
        - **TypeError** - 如果 `indices` 中的Tensor的dtype不是int32或者int64。
        - **TypeError** - 如果 `indices` 中的Tensor的dtype是不一致的。
        - **TypeError** - 如果 `accumulate` 的dtype不是bool。
        - **ValueError** - 如果 `values` 的秩不是1-D。
        - **ValueError** - 当Tensor本身的rank与 `indices` 的size相等时，如果 `values` 的size不为1
          或者不为 `indices` 中Tensor的最大size。
        - **ValueError** - 当Tensor本身的rank大于 `indices` 的size时，如果 `values` 的size不为1
          或者不为Tensor本身的最后一维的shape。
        - **ValueError** - 如果 `indices` 中的Tensor的秩不是1-D。
        - **ValueError** - 如果 `indices` 中的Tensor不是可广播的。
        - **ValueError** - 如果 `indices` 的size大于Tensor本身的秩。
