mindspore.ops.ComputeAccidentalHits
=====================================

.. py:class:: mindspore.ops.ComputeAccidentalHits(num_true=1)

    计算与目标类完全匹配的抽样样本的位置id。

    当目标类与抽样类匹配时，我们称之为"accidental hit"。accidental hit的计算结果包含三部分(indices、ids、weights)，其中index代表目标类中的行号，id代表候选抽样中的位置，weight为float类型中的最大值。

    参数：
        - **num_true** (int) - 每个训练样本的目标类数。默认值：1。

    输入：
        - **true_classes** (Tensor) - 目标类。数据类型为int32或int64，shape为 :math:`(batch\_size, num\_true)` 。
        - **sampled_candidates** (Tensor) - 算子的候选采样结果，代表训练样本的类别。其数据类型为int32或int64，shape为 :math:`(num\_sampled, )` 。

    输出：
        3个Tensor组成的元组。

        - **indices** (Tensor) - shape为 :math:`(num\_accidental\_hits, )` 的Tensor，具有与 `true_classes` 相同的类型。
        - **ids** (Tensor) - shape为 :math:`(num\_accidental\_hits, )` 的Tensor，具有与 `true_classes` 相同的类型。
        - **weights** (Tensor) - shape为 :math:`(num\_accidental\_hits, )` 的Tensor，类型为float32。

    异常：
        - **TypeError** - `num_true` 的数据类型不为int。
        - **TypeError** - `true_classes` 或 `sampled_candidates` 不是Tensor。
        - **TypeError** - `true_classes` 或 `sampled_candidates` 的数据类型既不是int32也不是int64。