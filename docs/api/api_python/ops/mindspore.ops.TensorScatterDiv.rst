mindspore.ops.TensorScatterDiv
==============================

.. py:class:: mindspore.ops.TensorScatterDiv

    根据指定的更新值和输入索引，进行除法运算更新输入Tensor的值。当同一索引有不同更新值时，更新的结果将是累积除法的结果。此操作更新后的结果是通过算子output返回，而不是直接原地更新input。

    更多参考相见 :func:`mindspore.ops.tensor_scatter_div`。
