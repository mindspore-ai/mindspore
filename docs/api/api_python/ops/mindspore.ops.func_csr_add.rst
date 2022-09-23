mindspore.ops.csr_add
=================================

.. py:function:: mindspore.ops.csr_add(a, b, alpha, beta)

    返回alpha * csr_a + beta * csr_b的结果，其中csr_a和csr_b是CSRTensor，alpha和beta是Tensor。

    参数：
        - **a** (CSRTensor) - 稀疏的 CSRTensor。
        - **b** (CSRTensor) - 稀疏的 CSRTensor。
        - **alpha** (Tensor) - 稠密张量，shape必须可以广播到 `a`。
        - **beta** (Tensor) - 稠密张量，shape必须可以广播到 `b`。

    返回：
        返回一个包含以下数据的CSRTensor。

        - **indptr** - 指示每行中非零值的起始点和结束点。
        - **indices** - 输入中所有非零值的列位置。
        - **values** - 稠密张量的非零值。
        - **shape** - csr_tensor 的形状。