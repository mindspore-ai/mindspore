mindspore.ops.csr_softmax
=================================

.. py:function:: mindspore.ops.csr_softmax(logits, dtype)

    计算 CSRTensorMatrix 的 softmax 。

    参数：
        - **logits** (CSRTensor) - 输入稀疏的 CSRTensor。
        - **dtype** (dtype) - 输入的数据类型。

    返回：
        - **CSRTensor** （CSRTensor） - 一个 csr_tensor 包含

          - **indptr** - 指示每行中非零值的起始点和结束点。
          - **indices** - 输入中所有非零值的列位置。
          - **values** - 稠密张量的非零值。
          - **shape** - csrtensor 的形状。
   