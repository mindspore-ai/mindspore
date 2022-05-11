mindspore.ops.TensorScatterSub
===============================

.. py:class:: mindspore.ops.TensorScatterSub

    根据指定的更新值和输入索引，通过减法运算更新输入Tensor的值。当同一个索引有多个不同值时，更新的结果将分别减去这些值。此操作几乎等同于使用 :class:`mindspore.ops.ScatterNdSub` ，只是更新后的结果是通过算子output返回，而不是直接原地更新input。

    更多参考详见 ：func:`mindspore.ops.tensor_scatter_sub`。
    