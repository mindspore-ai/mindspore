mindspore.ops.IndexFill
=======================

.. py:class:: mindspore.ops.IndexFill

    按 `index` 中给定的顺序选择索引，将输入Tensor `x` 的 `dim` 维度下的元素用 `value` 的值填充。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.index_fill`。

    输入：
        - **x** (Tensor) - 输入Tensor。
        - **dim** (Union[int, Tensor]) - 填充输入Tensor的维度，要求是一个int或者数据类型为int32或int64的零维Tensor。
        - **index** (Tensor) - 填充输入Tensor的索引，数据类型为int32。
        - **value** (Union[bool, int, float, Tensor]) - 填充输入Tensor的值。

    输出：
        填充后的Tensor。shape和数据类型与输入 `x` 相同。
