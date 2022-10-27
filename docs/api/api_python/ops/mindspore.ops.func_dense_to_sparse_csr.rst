mindspore.ops.dense_to_sparse_csr
=================================

.. py:function:: mindspore.ops.dense_to_sparse_csr(tensor: Tensor)

    将常规Tensor转为稀疏化的CSRTensor。

    .. note::
        现在只支持2维Tensor。

    参数：
        - **tensor** (Tensor) - 一个稠密Tensor，必须是2维。

    返回：
        返回一个2维的CSRTensor，是原稠密Tensor的稀疏化表示。其中数据分别为：

        - **indptr** (Tensor) - 一维整数张量，表示每行非零元素的在 `values` 中存储的起止位置。
        - **indices** (Tensor) - 一维整数张量，表示每个元素的列索引值。
        - **values** (Tensor) - 一维张量，用来表示索引对应的数值。
        - **shape** (tuple(int)) - 整数元组，用来指定稀疏矩阵的稠密形状。

    异常：
        - **TypeError** - `tensor` 不是Tensor。
        - **ValueError** - `tensor` 不是2维Tensor。
