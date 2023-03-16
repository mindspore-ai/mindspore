mindspore.ops.EditDistance
===========================

.. py:class:: mindspore.ops.EditDistance(normalize=True)

    计算Levenshtein编辑距离。它用于测量两个序列的相似性。输入是可变长度的序列，由SpaseTensors（hypothesis_indices, hypothesis_values, hypothesis_shape）和（truth_indices, truth_values, truth_shape）提供。

    .. math::
        \operatorname{lev}_{a, b}(i, j)=\left\{\begin{array}{ll}
        \max (i, j)  \qquad \qquad \qquad \qquad \qquad \quad \  \text { if } \min (i, j)=0 \\
        \min \left\{\begin{array}{ll}
        \operatorname{lev}_{a, b}(i-1, j)+1 & \\
        \operatorname{lev}_{a, b}(i, j-1)+1 & \text { otherwise. } \\
        \operatorname{lev}_{a, b}(i-1, j-1)+1_{\left(a_{i} \neq b_{j}\right)}
        \end{array}\right. &
        \end{array}\right.

    其中 :math:`a` 表示预测值， :math:`b` 表示真实值。为了便于理解，这里的i和j可以被视为a和b的长度。

    .. warning::
        - 如果输入 `truth_indices` 或者 `hypothesis_indices` 不是有序的， 可能会导致计算结果不符合预期， 建议调用该接口之前确保输入的稀疏张量 `truth_indices` 和 `hypothesis_indices` 都是升序排列的。

    参数：
        - **normalize** (bool) - 如果为True，则编辑距离将按真实值长度标准化。默认值：True。

    输入：
        - **hypothesis_indices** (Tensor) - 预测列表的索引。类型为Tensor，数据类型为int64，其shape为 :math:`(N, R)` 。
        - **hypothesis_values** (Tensor) - 预测列表的值。类型为Tensor，必须是长度为N的一维向量。
        - **hypothesis_shape** (Tensor) - 预测列表的shape。类型为Tensor，必须是长度为R的向量，数据类型为int64。只能是常量。
        - **truth_indices** (Tensor) - 真实列表的索引。类型为Tensor，数据类型为int64，其shape为 :math:`(M, R)` 。
        - **truth_values** (Tensor) - 真实列表的值。类型为Tensor，必须是长度为M的一维向量。
        - **truth_shape** (Tensor) - 真实列表的shape。类型为Tensor，必须是长度为R的向量，数据类型为int64。只能是常量。

    输出：
        Tensor，其秩为 `R-1` ，数据类型为float32。

    异常：
        - **TypeError** - 如果 `normalize` 不是bool。
