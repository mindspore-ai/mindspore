mindspore.ops.DiagPart
======================

.. py:class:: mindspore.ops.DiagPart

    提取输入Tensor的对角线元素。

    假如 `input_x` 有维度 :math:`[D_1,..., D_k, D_1,..., D_k]`，那么输出是一个秩为k的Tensor，维度为 :math:`[D_1,..., D_k]`，其中：

    :math:`output[i_1,..., i_k] = input_x[i_1,..., i_k, i_1,..., i_k]`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。它的秩为2k(k > 0)。

    输出：
        Tensor，与 `input` 有相同的数据类型。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果 `input` 的秩不是偶数，或为零。
        - **ValueError** - 如果 `input` 的shape不满足：`input_shape[i] == input_shape[i + len(input_shape)/2]`。
