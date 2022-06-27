mindspore.ops.TensorScatterElements
===================================

.. py:class:: mindspore.ops.TensorScatterElements

    根据指定的更新值和输入索引，通过reduction操作更新输入Tensor的值。

    .. warning::
    如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。

    更多参考详见 :func:`mindspore.ops.tensor_scatter_elements`。
