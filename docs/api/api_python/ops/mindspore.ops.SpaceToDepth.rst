mindspore.ops.SpaceToDepth
==========================

.. py:class:: mindspore.ops.SpaceToDepth(block_size)

    将空间维度分块，增加Tensor深度。

    输出Tensor的高度为 :math:`height / block\_size`。

    输出Tensor的宽度为 :math:`weight / block\_size`。

    输出Tensor的深度为 :math:`block\_size * block\_size * input\_depth`。

    输入Tensor的高度和宽度必须可被 `block_size` 整除。格式为"NCHW"（batch_size，深度，高度，宽度）。

    参数：
        - **block_size** (int) - 用于划分空间维度的子块的大小。必须>=2。

    输入：
        - **x** (Tensor) - 四维Tensor。数据类型为Number。

    输出：
        四维Tensor，数据类型与 `x` 相同，shape： :math:`(N, (C_{in} * \text{block_size} * 2), H_{in} / \text{block_size}, W_{in} / \text{block_size})` 。

    异常：
        - **TypeError** - `block_size` 不是int类型。
        - **ValueError** - `block_size` 小于2。
        - **ValueError** - `x` 的维度不为4。
