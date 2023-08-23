mindspore.ops.ExpandDims
========================

.. py:class:: mindspore.ops.ExpandDims

    对输入 `input_x` 在给定的轴上添加额外维度， `input_x` 的维度应该大于等于1。

    获取更多详情请查看 :func:`mindspore.ops.expand_dims` 。

    输入：
        - **input_x** (Tensor) - Tensor的shape是 :math:`(x_1, x_2, ..., x_R)` 。
        - **axis** (int) - 指定需添加额外维度的轴。轴值必须在 `[-input_x.ndim-1, input_x.ndim]` 范围内。只支持常量值。

    输出：
        Tensor，Tensor的shape为 :math:`(1, x_1, x_2, ..., x_R)` （ `axis` 的值为0）。其数据类型与 `input_x` 的相同。
