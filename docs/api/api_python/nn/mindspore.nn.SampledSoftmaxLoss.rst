mindspore.nn.SampledSoftmaxLoss
================================

.. py:class:: mindspore.nn.SampledSoftmaxLoss(num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, seed=0, reduction='none')

    抽样交叉熵损失函数。

    一般在类别数很大时使用。可加速训练以交叉熵为损失函数的分类器。

    参数：
        - **num_sampled** (int) - 抽样的类别数。
        - **num_classes** (int) - 类别总数。
        - **num_true** (int) - 每个训练样本的类别数。默认值：1。
        - **sampled_values** (Union[list, tuple]) - 抽样候选值。由 `*CandidateSampler` 函数返回的(`sampled_candidates`, `true_expected_count` , `sampled_expected_count`)的list或tuple。如果默认值为None，则应用 `UniformCandidateSampler` 。
        - **remove_accidental_hits** (bool) - 是否移除抽样中的目标类等于标签的情况。默认值：True。
        - **seed** (int) - 抽样的随机种子。默认值：0。
        - **reduction** (str) - 指定应用于输出结果的计算方式。取值为"mean"，"sum"，或"none"。取值为"none"，则不执行reduction。默认值："none"。

    输入：
        - **weights** (Tensor) - 输入的权重，shape为 :math:`(C, dim)` 的Tensor。
        - **bias** (Tensor) - 分类的偏置。shape为 :math:`(C,)` 的Tensor。
        - **labels** (Tensor) - 输入目标值Tensor，其shape为 :math:`(N, num\_true)` ，其数据类型为 `int64, int32` 。
        - **logits** (Tensor) - 输入预测值Tensor，其shape为 :math:`(N, dim)` 。

    输出：
        Tensor或Scalar，如果 `reduction` 为'none'，则输出是shape为 :math:`(N,)` 的Tensor。否则，输出为Scalar。

    异常：
        - **TypeError** - `sampled_values` 不是list或tuple。
        - **TypeError** - `labels` 的数据类型既不是int32，也不是int64。
        - **ValueError** - `reduction` 不为'none'、'mean'或'sum'。
        - **ValueError** - `num_sampled` 或 `num_true` 大于 `num_classes` 。
        - **ValueError** - `sampled_values` 的长度不等于3。
