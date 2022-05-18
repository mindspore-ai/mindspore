mindspore.nn.SparseToDense
===========================

.. py:class:: mindspore.nn.SparseToDense

    将稀疏Tensor转换为稠密Tensor。

    在Python中，为了方便使用，三个Tensor被收集到一个SparseTensor类中。MindSpore使用三个独立的稠密Tensor： `indices` 、 `values` 和 `dense_shape` 来表示稀疏Tensor。在调用 :class:`mindspore.ops.SparseToDense` 之前，可以单独的将 `indices` 、 `values` 和 `dense_shape` 传递给稀疏Tensor对象。

    **输入：**
    
    - **coo_tensor** (:class:`mindspore.COOTensor`) - 要转换的稀疏Tensor。

    **输出：**

    Tensor，由稀疏Tensor转换而成。

    **异常：**

    - **TypeError** - `sparse_tensor.indices` 不是Tensor。
    - **TypeError** - `sparse_tensor.values` 不是Tensor。
    - **TypeError** - `sparse_tensor.dense_shape` 不是tuple。