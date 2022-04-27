mindspore.nn.MatrixDiag
========================

.. py:class:: mindspore.nn.MatrixDiag

    根据对角线值返回一批对角矩阵。

    假设 `x` 有 :math:`k` 个维度 :math:`[I, J, K, ..., N]` ，则输出秩为 :math:`k+1` 且维度为 :math:`[I, J, K, ..., N, N]` 的Tensor，其中： :math:`output[i, j, k, ..., m, n] = 1\{m=n\} * x[i, j, k, ..., n]` 。

    **输入：**
    
    - **x** (Tensor) - 输入任意维度的对角线值。支持的数据类型包括：float32、float16、int32、int8和uint8。

    **输出：**
    
    Tensor，shape与输入 `x` 相同。Shape必须为 :math:`x.shape + (x.shape[-1], )` 。

    **异常：**
    
    - **TypeError** - `x` 的数据类型不是float32、float16、int32、int8或uint8。