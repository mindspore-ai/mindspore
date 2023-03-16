mindspore.ops.RNNTLoss
=======================

.. py:class:: mindspore.ops.RNNTLoss(blank_label=0)

    计算相对于softmax输出的RNNTLoss及其梯度。

    参数：
        - **blank_label** (int) - 空白标签。默认值：0。

    输入：
        - **acts** (Tensor) - Tensor，shape为 :math:`(B, T, U, V)` 。数据类型必须为float16或float32。
        - **labels** (Tensor) - Tensor，shape为 :math:`(B, U-1)` 。数据类型为int32。
        - **input_lengths** (Tensor) - Tensor，shape为 :math:`(B,)` 。数据类型为int32。
        - **label_lengths** (Tensor) - Tensor，shape为 :math:`(B,)` 。数据类型为int32。

    输出：
        - **costs** (Tensor) - Tensor，shape为 :math:`(B,)` 。数据类型为int32。
        - **grads** (Tensor) - 具有与 `acts` 相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `acts` 、 `labels` 、 `input_lengths` 或 `label_lengths` 不是Tensor。
        - **TypeError** - 如果 `acts` 的dtype既不是float16也不是float32。
        - **TypeError** - 如果 `labels`、 `input_lengths` 或 `label_lengths` 的dtype不是int32。
