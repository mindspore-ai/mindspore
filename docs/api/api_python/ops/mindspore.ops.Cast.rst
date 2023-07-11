mindspore.ops.Cast
===================

.. py:class:: mindspore.ops.Cast

    转换输入Tensor的数据类型。

    .. note::
        将复数转换为bool类型的时候，不考虑复数的虚部，只要实部不为零， 返回True， 否则返回Fasle。

    输入：
        - **input_x** (Union[Tensor, Number]) - 输入要进行数据类型转换的Tensor，其shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **type** (dtype.Number) - 指定转换的数据类型。仅支持常量值。

    输出：
        Tensor，其shape与 `input_x` 相同，即 :math:`(x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - `input_x` 既不是Tensor也不是数值型。
        - **TypeError** - `type` 不是数值型。