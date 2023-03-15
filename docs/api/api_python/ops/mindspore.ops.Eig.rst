mindspore.ops.Eig
==================

.. py:class:: mindspore.ops.Eig(compute_v=False)

    计算输入方阵（batch方阵）的特征值和特征向量。

    参数：
        - **compute_v** (bool，可选) - 如果为True，同时计算特征值和特征向量，如果为False，只计算特征值，默认值：False。

    输入：
        - **x** (Tensor) - 方阵。shape为 :math:`(*, N, N)`，数据类型支持
          float32、float64、complex64、complex128。

    输出：
        - **eigen_values** (Tensor) - shape为 :math:`(*, N)`，其中的每个向量代表对应矩阵的特征值，特征值之间没有顺序关系。
        - **eigen_vectors** (Tensor) - 如果 `compute_v` 为False，此为空Tensor，否则，为shape :math:`(*, N, N)` 的Tensor。
          其列表示相应特征值的规范化（单位长度）特征向量。

    异常：
        - **TypeError** - 如果 `compute_v` 数据类型不是一个bool。
        - **TypeError** - 如果 `x` 的数据类型不是： float32、float64、complex64或者complex128。
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **ValueError** - 如果 `x` 不是一个方阵（batch方阵）。
