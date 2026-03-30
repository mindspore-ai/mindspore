mindspore.ops.normal
======================

.. py:function:: mindspore.ops.normal(shape, mean, stddev, seed=None)

    返回符合正态（高斯）分布的随机tensor。

    .. warning::
        - 2.9.0（不含）之后版本非兼容性变更：接口将改为按 `mean/std/size` 生成，不再接受 `shape`。
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **shape** (tuple) - 指定shape。
        - **mean** (Union[Tensor, int, float]) - 返回tensor的正态分布平均值。
        - **stddev** (Union[Tensor, int, float]) - 返回tensor的正态分布标准差。
        - **seed** (int，可选) - 随机种子，默认 ``None`` ，等同于 ``0`` 。

    返回：
        Tensor
