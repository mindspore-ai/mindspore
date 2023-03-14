mindspore.ops.eig
==================

.. py:function:: mindspore.ops.eig(A)

    计算输入方阵（batch方阵）的特征值和特征向量。

    参数：
        - **A** (Tensor) - 方阵。shape为 :math:`(*, N, N)`，数据类型支持
          float32、float64、complex64、complex128。

    返回：
        - **eigen_values** (Tensor) - shape为 :math:`(*, N)`，其中的每个向量代表对应矩阵的特征值，特征值之间没有顺序关系。
        - **eigen_vectors** (Tensor) - 如果 `compute_v` 为False，此为空Tensor，否则，为shape :math:`(*, N, N)` 的Tensor。
          其列表示相应特征值的规范化（单位长度）特征向量。

    异常：
        - **TypeError** - 如果 `A` 的数据类型不是： float32、float64、complex64或者complex128。
        - **TypeError** - 如果 `A` 不是一个Tensor。
        - **ValueError** - 如果 `A` 不是一个方阵（batch方阵）。
