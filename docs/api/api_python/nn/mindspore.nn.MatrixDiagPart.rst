mindspore.nn.MatrixDiagPart
============================

.. py:class:: mindspore.nn.MatrixDiagPart

    返回批对角矩阵的对角线部分。

    假设 `x` 有 :math:`k` 个维度 :math:`[I, J, K, ..., M, N]` ，则输出秩为 :math:`k-1` 且维度为 :math:`[I, J, K, ..., min(M, N)]` 的Tensor，其中： :math:`output[i, j, k, ..., n] = x[i, j, k, ..., n, n]` 。

    **输入：**
    
    - **x** (Tensor) - 输入具有批对角的Tensor。支持的数据类型包括：float32、float16、int32、int8和uint8。

    **输出：**
    
    Tensor，shape与输入 `x` 相同。shape必须为 :math:`x.shape[:-2]+[min(x.shape[-2:])]` 。

    **异常：**
    
    - **TypeError** - `x` 的数据类型不是float32、float16、int32、int8或uint8。