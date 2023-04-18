mindspore.ops.Reshape
======================

.. py:class:: mindspore.ops.Reshape

    基于给定的shape，对输入Tensor进行重新排列。

    更多参考详见 :func:`mindspore.ops.reshape`。

    输入：
        - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **input_shape** (tuple[int]) - 给定的shape，由多个整数构成，如 :math:`(y_1, y_2, ..., y_S)` 。

    输出：
        Tensor，其shape为 :math:`(y_1, y_2, ..., y_S)` 。
