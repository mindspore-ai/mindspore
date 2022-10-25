mindspore.ops.dense_to_sparse_coo
=================================

.. py:function:: mindspore.ops.dense_to_sparse_coo(tensor: Tensor)

    将常规Tensor转为稀疏化的COOTensor。

    .. note::
        现在只支持2维Tensor。

    参数：
        - **tensor** (Tensor) - 一个稠密Tensor，必须是2维。

    返回：
        返回一个2维的COOTensor，是原稠密Tensor的稀疏化表示。分为：

        - **indices** (Tensor) - 二维整数张量，表示稀疏张量中 `values` 所处的位置索引。
        - **values** (Tensor) - 一维张量，用来给 `indices` 中的每个元素提供数值。
        - **shape** (tuple(int)) - 整数元组，用来指定稀疏矩阵的稠密形状。


    异常：
        - **TypeError** - `tensor` 不是Tensor。
        - **ValueError** - `tensor` 不是2维Tensor。
