mindspore.ops.CTCLoss
=====================

.. py:class:: mindspore.ops.CTCLoss(preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False)

    计算CTC(Connectionist Temporal Classification)损失和梯度。

    该接口的底层调用了第三方baidu-research::warp-ctc的实现。CTC算法是在 `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with Recurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 中提出的。 

    CTCLoss计算连续时间序列和目标序列之间的损失。 CTCLoss对输入到目标的概率求和，产生一个损失值，该值相对于每个输入节点是可微的。假设输入与目标的对齐是“多对一”的，这样目标序列的长度必须小于或等于输入的长度。 

    参数：
        - **preprocess_collapse_repeated** (bool) - 如果为True，在CTC计算之前将折叠重复标签。默认值：False。
        - **ctc_merge_repeated** (bool) - 如果为False，在CTC计算过程中，重复的非空白标签不会被合并，这些标签将被解释为单独的标签。这是CTC的简化版本。默认值：True。
        - **ignore_longer_outputs_than_inputs** (bool) - 如果为True，则输出比输入长的序列将被忽略。默认值：False。

    输入：
        - **x** (Tensor) - 输入Tensor，其shape为 :math:`(max\_time, batch\_size, num\_classes)` 的三维Tensor。 `num_classes` 表示类别数，必须是 `num_labels + 1` ， `num_labels` 表示实际标签的数量。保留空白标签。默认空白标签为 `num_classes - 1` 。数据类型必须为float16、float32或float64。
        - **labels_indices** (Tensor) - 标签的索引。 `labels_indices[i, :] = [b, t]` 表示 `labels_values[i]` 存储 `(batch b, time t)` 的ID。数据类型必须为int64，秩必须为2。
        - **labels_values** (Tensor) - 一维Tensor。这些值与给定的batch size和时间相关联。数据类型必须为int32。 `labels_values[i]` 必须在 `[0, num_classes)` 的范围内。
        - **sequence_length** (Tensor) - 包含序列长度的Tensor，shape为 :math:`(batch\_size, )` 。数据类型必须为int32。Tensor中的每个值不得大于最大时间。

    输出：
        - **loss** (Tensor) - 包含对数概率的Tensor，shape为 :math:`(batch\_size, )` 。Tensor的数据类型与 `x` 相同。
        - **gradient** (Tensor) - `loss` 的梯度，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `preprocess_collapse_repeated` 、 `ctc_merge_repeated` 或 `ignore_longer_outputs_than_inputs` 不是bool。
        - **TypeError** - `x` 、 `labels_indices` 、 `labels_values` 或 `sequence_length` 不是Tensor。
        - **ValueError** - `labels_indices` 的秩不等于2。
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。
        - **TypeError** - `labels_indices` 的数据类型不是int64。
        - **TypeError** - `labels_values` 或 `sequence_length` 的数据类型不是int32。
