mindspore.ops.random_gamma
==========================

.. py:function:: mindspore.ops.random_gamma(shape, alpha, seed=None)

    `mindspore.ops.random_gamma` 从2.10.0版本开始已被弃用，并将在未来版本中被移除。

    根据伽马分布生成随机数。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：该采样接口将改为分布对象形式，
        届时请使用对应的分布类代替。

    参数：
        - **shape** (Tensor) - 指定生成随机数的shape。
        - **alpha** (Tensor) - :math:`\alpha` 分布的参数。必须是非负数。
        - **seed** (int, 可选) - 随机种子，必须为非负数。默认 ``None`` 。

    返回：
        Tensor。形状为 `mindspore.ops.concat([shape, alpha.shape], axis=0)` 。数据类型和alpha一致。
