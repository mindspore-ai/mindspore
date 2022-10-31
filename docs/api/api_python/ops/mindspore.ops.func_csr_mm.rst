mindspore.ops.csr_mm
=================================

.. py:function:: mindspore.ops.csr_mm(a: CSRTensor, b: CSRTensor or Tensor, trans_a: bool, trans_b: bool, adjoint_a: bool, adjoint_b: bool)

    返回稀疏矩阵a与稀疏矩阵或稠密矩阵b的矩阵乘法结果。

    .. note::
        若右矩阵为Tensor，则仅支持安装了LLVM12.0.1的CPU后端或GPU后端。
        若右矩阵为CSRTensor， 则仅支持GPU后端。

    参数：
        - **a** (CSRTensor) - 稀疏的 CSRTensor。
        - **b** (CSRTensor 或 Tensor) - 稀疏的 CSRTensor或稠密矩阵。
        - **trans_a** (Tensor) - 是否对矩阵a进行转置。
        - **trans_b** (Tensor) - 是否对矩阵b进行转置。
        - **adjoint_a** (Tensor) - 是否对矩阵a进行共轭。
        - **adjoint_b** (Tensor) - 是否对矩阵b进行共轭。

    返回：
        返回稀疏矩阵，类型为CSRTensor。