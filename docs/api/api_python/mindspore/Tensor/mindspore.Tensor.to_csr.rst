mindspore.Tensor.to_csr
=======================

.. py:method:: mindspore.Tensor.to_csr()

    将常规Tensor转为稀疏化的CSRTensor。

    .. note::
        现在只支持二维Tensor。

    返回：
        返回一个二维的CSRTensor，是原稠密Tensor的稀疏化表示。其中数据分别为：

        - **indptr** (Tensor) - 一维整数Tensor，表示每行非零元素在 `values` 中存储的起止位置。
        - **indices** (Tensor) - 一维整数Tensor，表示每个元素的列索引值。
        - **values** (Tensor) - 一维Tensor，用来表示索引对应的数值。
        - **shape** (tuple(int)) - 整数元组，用来指定稀疏矩阵的稠密形状。

    异常：
        - **ValueError** - Tensor的shape不是二维。