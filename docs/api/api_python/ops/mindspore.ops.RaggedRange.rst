mindspore.ops.RaggedRange
==========================

.. py:class:: mindspore.ops.RaggedRange(Tsplits)

    返回包含指定数数列的RaggedTensor。

    参数：
        - **Tsplits** (mindspore.dtype) - 输出的类型。它的值必须是mstype.int32或者mstype.int64。

    输入：
        - **starts** (Tensor) - 每个数列的开始。是一个 0D或1D Tensor，数据类型为int32、int64、float32或float64。
        - **limits** (Tensor) - 每个数列的上限，shape与数据类型与 `starts` 一致。
        - **deltas** (Tensor) - 每个数列增量，shape与数据类型与 `starts` 一致，其中所有元素的值不能为0。

    输出：
        - **rt_nested_splits** (Tensor) - 返回RagdTensor的嵌套拆分Tensor，数据类型类型为 `Tsplits` 。shape等于输入 `starts` 的shape加1。
        - **rt_dense_values** (Tensor) - 返回RagdTensor的密集值Tensor，其数据类型与输入 `starts` 相同。设输入 `starts、` `limits` 和 `delta` 的大小为i。
          - 如果 `starts` 、 `limits` 和 `delta` 的数据类型为int32或int64，则输出 `rt_dense_values` 的shape等于 :math:`sum(abs(limits[i] - starts[i]) + abs(deltas[i] - 1) / abs(deltas[i]))` 。
          - 如果 `starts` 、 `limits` 和 `delta` 的数据类型为float32或者float64，则输出 `rt_dense_values` 的shape等于 :math:`sum(ceil(abs((limits[i] - starts[i]) / deltas[i]))` 。

    异常：
        - **TypeError** - 如任意一个输入不是Tensor。
        - **TypeError** - 如果  `starts` 的数据类型不是：int32、int64、float32或float64。
        - **TypeError** - 如果 `starts` 、 `limits` 和 `deltas` 的数据类型不一致。
        - **TypeError** - 如果 `Tsplits` 不是mstype.int32或者mstype.int64。
        - **ValueError** - 如果 `starts` 、 `limits` 和 `deltas` 不是 0D或1D Tensor。
        - **ValueError** - 如果 `deltas` 等于0。
        - **ValueError** - 如果 `starts` 、 `limits` 和 `deltas` 的shape不一致。
