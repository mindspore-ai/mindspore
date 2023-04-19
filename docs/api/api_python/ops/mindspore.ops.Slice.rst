mindspore.ops.Slice
====================

.. py:class:: mindspore.ops.Slice

    根据指定shape对输入Tensor进行切片。

    更多参考详见 :func:`mindspore.ops.slice`。

    输入：
        - **input_x** (Tensor) - Slice的输入，任意维度的Tensor。其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **begin** (Union[tuple, list]) - 切片的起始位置。只支持常量值（>=0）。
        - **size** (Union[tuple, list]) - 切片的大小。只支持常量值。

    输出：
        Tensor，shape与输入 `size` 相同，数据类型与输入 `input_x` 的相同。
