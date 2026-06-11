mindspore.ops.gamma
====================

.. py:function:: mindspore.ops.gamma(shape, alpha, beta, seed=None)

    `mindspore.ops.gamma` 从2.10.0版本开始已被弃用，并将在未来版本中被移除。

    根据伽马分布生成随机数。

    支持广播。

    .. warning::
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。
        - 从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：该采样接口将改为分布对象形式，
          届时请使用对应的分布类代替。

    参数：
        - **shape** (tuple) - 指定shape。
        - **alpha** (Tensor) - shape参数。
        - **beta** (Tensor) - 逆尺度参数。
        - **seed** (int，可选) - 随机种子，默认 ``None`` 。

    返回：
        Tensor
