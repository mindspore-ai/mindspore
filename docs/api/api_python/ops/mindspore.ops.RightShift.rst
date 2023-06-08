mindspore.ops.RightShift
=========================

.. py:class:: mindspore.ops.RightShift

    将Tensor `input_x` 的每个元素右移 Tensor `input_y` 中对应位数。输入为两个Tensor，数据类型需要保持一致，他们之间的shape可以广播。

    .. math::

        \begin{aligned}
        &out_{i} =x_{i} >> y_{i}
        \end{aligned}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **input_x** (Tensor) - 被操作Tensor，将被逐元素位右移 `input_y` 位。支持所有int和uint类型。
        - **input_y** (Tensor) - 右移位数。数据类型必须和 `input_x` 一致。

    输出：
        - **output** (Tensor) - 输出Tensor，据类型和 `input_x` 一致。

    异常：
        - **TypeError** - `input_x` 或者 `input_y` 不是Tensor。
        - **TypeError** - `input_x` 和 `input_y` 不能发生广播。
