mindspore.ops.DepthToSpace
===========================

.. py:class:: mindspore.ops.DepthToSpace(block_size)

    将深度数据重新排列到空间维度中。

    这是SpaceToDepth的反向操作。

    输出Tensor的深度为 :math:`input\_depth / (block\_size * block\_size)` 。

    输出Tensor的 `height` 维度为 :math:`height * block\_size` 。

    输出Tensor的 `weight` 维度为 :math:`weight * block\_size` 。

    输入Tensor的深度必须可被 `block_size * block_size` 整除。数据格式为"NCHW"。

    参数：
        - **block_size** (int) - 用于划分深度数据的块大小。其值必须>=2。

    输入：
        - **x** (Tensor) - 输入Tensor。它必须为四维，其shape为 :math:`(N, C_{in}, H_{in}, W_{in})` ，数据类型为数值型。

    输出：
        Tensor，shape为 :math:`(N, C_{in} / \text{block_size} ^ 2, H_{in} * \text{block_size},
        W_{in} * \text{block_size})` 。

    异常：
        - **TypeError** - 如果 `block_size` 不是int。
        - **ValueError** - 如果 `block_size` 小于2。
        - **ValueError** - 如果 `x` 的shape长度不等于4。
