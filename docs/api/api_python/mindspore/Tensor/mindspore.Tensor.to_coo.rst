mindspore.Tensor.to_coo
=======================

.. py:method:: mindspore.Tensor.to_coo()

    将常规Tensor转为稀疏化的COOTensor。

    .. note::
        现在只支持二维Tensor。

    返回：
        返回一个二维的COOTensor，是原稠密Tensor的稀疏化表示。其中数据分别为：

        - **indices** (Tensor) - 二维整数张量，表示稀疏张量中 `values` 所处的位置索引。
        - **values** (Tensor) - 一维张量，用来给 `indices` 中的每个元素提供数值。
        - **shape** (tuple(int)) - 整数元组，用来指定稀疏矩阵的稠密形状。

    异常：
        - **ValueError** - Tensor的shape不是二维。