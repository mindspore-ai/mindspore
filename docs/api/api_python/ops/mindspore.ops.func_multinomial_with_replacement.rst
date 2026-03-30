mindspore.ops.multinomial_with_replacement
==========================================

.. py:function:: mindspore.ops.multinomial_with_replacement(x, seed, offset, numsamples, replacement=False)

    生成一个多项式分布的tensor。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - 输入的行不需要求和为1（在这种情况下，使用值作为权重），但必须是非负的、有限的，并且具有非零和。
        - `seed` 如果为 ``-1`` 且 `offset` 为 ``0`` ，则随机数生成器将使用随机种子。

    参数：
        - **x** (Tensor) - 一维或二维的输入tensor，包含概率的累积和。
        - **seed** (int) - 随机种子。
        - **offset** (int) - 偏移量。
        - **numsamples** (int) - 采样的次数。
        - **replacement** (bool，可选) - 是否放回。默认 ``False`` 。

    返回：
        Tensor
