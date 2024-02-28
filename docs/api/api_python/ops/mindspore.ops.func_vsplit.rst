mindspore.ops.vsplit
=====================

.. py:function:: mindspore.ops.vsplit(input, indices_or_sections)

    根据 `indices_or_sections` 将输入Tensor `input` 垂直分割成多个子Tensor。等同于 :math:`axis=0` 时的 `ops.tensor_split` 。

    参数：
        - **input** (Tensor) - 待分割的Tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 参考 :func:`mindspore.ops.tensor_split`.

    返回：
        tuple[Tensor]。
