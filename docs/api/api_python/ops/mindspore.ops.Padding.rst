mindspore.ops.Padding
=====================

.. py:class:: mindspore.ops.Padding(pad_dim_size=8)

    将输入Tensor的最后一个维度从1扩展到 `pad_dim_size` ，其填充值为0。

    更多参考详见 :func:`mindspore.ops.padding`。

    参数：
        - **pad_dim_size** (int，可选) - 指定填充的大小，待扩展的 `x` 的最后一个维度的值，必须为正数。默认值： ``8`` 。

    输入：
        - **x** (Tensor) - 输入Tensor，二维或者更高维Tensor， `x` 的最后一个维度必须为1。数据类型为Number。

    输出：
        填充后的Tensor。
