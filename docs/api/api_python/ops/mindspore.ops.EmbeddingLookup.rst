mindspore.ops.EmbeddingLookup
===============================

.. py:class:: mindspore.ops.EmbeddingLookup

    根据指定的索引，返回输入Tensor的切片。

    此算子在 `axis = 0` 上的运行与GatherV2的功能相似，只是多一个 `offset` 输入。

    输入：
        - **input_params** (Tensor) - shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。是一个Tensor切片。当前，只支持二维。
        - **input_indices** (Tensor) - shape为 :math:`(y_1, y_2, ..., y_S)` 的Tensor。指定输入Tensor元素的索引。当取值超出  `input_params` 在该维度的最大长度时，超出部分将返回0值。不支持负值，否则结果将未定义。其数据类型为int32或int64。
        - **offset** (int) - 指定 `input_params` 切片的偏移值。实际索引等于 `input_indices` 减去 `offset` 。

    输出：
        Tensor，shape为 :math:`(z_1, z_2, ..., z_N)` 的Tensor。数据类型与 `input_params` 相同。

    异常：
        - **TypeError** - `input_indices` 的数据类型不是int。
        - **ValueError** - `input_params` 的shape长度大于2。