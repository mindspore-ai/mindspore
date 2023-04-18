mindspore.ops.TupleToArray
===========================

.. py:class:: mindspore.ops.TupleToArray

    将tuple转换为Tensor。

    更多参考详见 :func:`mindspore.ops.tuple_to_array`。

    输入：
        - **input_x** (tuple) - 数值型组成的tuple。其元素具有相同的类型。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor。如果输入tuple包含 `N` 个数值型元素，则输出Tensor的shape为 :math:`(N,)` 。
