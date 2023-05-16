mindspore.ops.Squeeze
=====================

.. py:class:: mindspore.ops.Squeeze(axis=())

    返回删除指定 `axis` 中大小为1的维度后的Tensor。

    更多参考详见 :func:`mindspore.ops.squeeze`。

    参数：
        - **axis** (Union[int, tuple(int)]) - 指定待删除shape的维度索引，它会删除给定 `axis` 参数中所有大小为1的维度。如果指定了维度索引，其数据类型必须为int32或int64。默认值： ``()`` 。

    输入：
        - **input_x** (Tensor) - 用于计算Squeeze的输入Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。

    输出：
        Tensor，shape为 :math:`(x_1, x_2, ..., x_S)` 。
