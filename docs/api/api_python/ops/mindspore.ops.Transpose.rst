mindspore.ops.Transpose
=======================

.. py:class:: mindspore.ops.Transpose

    根据指定的排列对输入的Tensor进行数据重排。

    更多参考详见 :func:`mindspore.ops.transpose`。

    输入：
        - **input_x** (Tensor) - 输入Tensor，其shape是 :math:`(x_1, x_2, ..., x_R)` 。
        - **input_perm** (tuple[int]) - 指定排列。 `input_perm` 中的元素由 `input_x` 的每个维度的索引组成。 `input_perm` 的长度和 `input_x` 的shape相同。只支持常量值。其范围在[0, rank(input_x))内。

    输出：
        Tensor，输出Tensor的数据类型与 `input_x` 相同，输出Tensor的shape由 `input_x` 的shape和 `input_perm` 的值决定。
