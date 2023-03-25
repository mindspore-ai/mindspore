mindspore.ops.TensorScatterElements
===================================

.. py:class:: mindspore.ops.TensorScatterElements(axis=0, reduction="none")

    根据指定的规约算法逐元素更新输入Tensor的值。

    更多参考相见 :func:`mindspore.ops.tensor_scatter_elements`。

    .. warning::
        如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。
