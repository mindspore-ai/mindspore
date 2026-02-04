mindspore.Tensor.index_put
==========================

.. py:method:: mindspore.Tensor.index_put(indices, values, accumulate=False)

    根据 `indices` 中的下标值，使用 `values` 中的数值替换Tensor `self` 中的相应元素的值。非原地更新版本的 :func:`mindspore.Tensor.index_put_` 。

    .. warning::
        以下场景将导致不可预测的行为：

        - 如果 `accumulate` 为 `False`，且 `indices` 存在重复元素。

    参数：
        - **indices** (tuple[Tensor], list[Tensor]) - 元素类型是int32或者int64，用于对 `self` 中的元素进行索引。
          `indices` 中的Tensor的秩应为1-D， `indices` 的size应小于或等于 `self` 的秩， `indices` 中的Tensor应是可广播的。
        - **values** (Tensor) - 一个一维的Tensor，数据类型与 `self` 相同。如果其size为1，则它是可广播的。
        - **accumulate** (bool，可选) - 如果 `accumulate` 为 ``True``， `values` 中的元素被累加到 `self` 的相应元素上；
          否则，用 `values` 中的元素取代 `self` 的相应元素。默认值: ``False`` 。

    返回：
        Tensor，其数据类型和shape与 `self` 相同。

    异常：
        - **TypeError** - 如果 `self` 的dtype与 `values` 的dtype不同。
        - **TypeError** - 如果 `indices` 的dtype不是tuple[Tensor]或者list[Tensor]。
        - **TypeError** - 如果 `indices` 中的Tensor的dtype不是int32或者int64。
        - **TypeError** - 如果 `indices` 中的Tensor的dtype是不一致的。
        - **TypeError** - 如果 `accumulate` 的dtype不是bool。
        - **ValueError** - 如果 `values` 的秩不是1-D。
        - **ValueError** - 当 `self` 的rank与 `indices` 的size相等时，如果 `values` 的size不为1\
          或者不为 `indices` 中Tensor的最大size。
        - **ValueError** - 当 `self` 的rank大于 `indices` 的size时，如果 `values` 的size不为1\
          或者不为 `self` 的最后一维的shape。
        - **ValueError** - 如果 `indices` 中的Tensor的秩不是1-D。
        - **ValueError** - 如果 `indices` 中的Tensor不是可广播的。
        - **ValueError** - 如果 `indices` 的size大于 `self` 的秩。
