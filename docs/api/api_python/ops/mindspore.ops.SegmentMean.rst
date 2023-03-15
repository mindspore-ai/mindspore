mindspore.ops.SegmentMean
==========================

.. py:class:: mindspore.ops.SegmentMean

    计算Tensor中每个分段的均值。

    具体来说，生成一个新的Tensor `output` ，满足 :math:`output_i=mean_j(input\_x_j)` ，其
    中均值在满足 :math:`segment\_ids[j] == i` 这个条件的所有的 :math:`j` 对应的元素中取得。
    如果一个分段中没有元素 :math:`i` ，则输出Tensor中相应的元素将被设置为零：:math:`output[i] = 0`。

    .. warning::
        如果 `input_x` 的数据类型是复数，则无法计算其梯度。

    输入：
        - **input_x** (Tensor) - 输入Tensor。数据类型为实数且秩不小于1的输入Tensor。
        - **segment_ids** (Tensor) - 数据类型为为int32或int64的一维Tensor。Tensor的大小必须等于 `input_x` 的shape的第一维。其值必须按升序排序，不需要覆盖所有有效值范围内的值，但必须是正整数。只允许常量值。

    输出：
        Tensor，其数据类型和shape维度与 `input_x` 相同。shape的第一个维度等于 `segment_ids` 最后一个元素的值加1，其他维度与 `input_x` 一致。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **TypeError** - 如果 `segment_ids` 不是Tensor。
        - **TypeError** - 如果 `input_x` 的数据类型不支持。
        - **TypeError** - 如果 `segment_ids` 的数据类型不支持。
        - **ValueError** - 如果 `input_x` 的秩小于1。
        - **ValueError** - 如果 `segment_ids` 的秩不等于1。
        - **ValueError** - 如果 `segment_ids` 的长度不等于 `input_x` shape第一维的大小。
        - **ValueError** - 如果 `segment_ids` 含有负数。
        - **ValueError** - 如果 `segment_ids` 不是升序排列。
