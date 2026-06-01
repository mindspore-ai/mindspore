mindspore.ops.random_poisson
============================

.. py:function:: mindspore.ops.random_poisson(shape, rate, seed=None, dtype=mstype.float32)

    `mindspore.ops.random_poisson` 从2.10.0版本开始已被弃用，并将在未来版本中被移除。

    从指定均值为 `rate` 的泊松分布中，生成指定shape的随机样本。

    .. math::

        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    .. warning::
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。
        - 从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：签名将调整为 ``poisson(input, generator=None)`` ，当前 `shape` 入参语义与新签名不兼容。

    参数：
        - **shape** (Tensor) - 指定生成随机数的shape。一维整型tensor。
        - **rate** (Tensor) - 泊松分布的 :math:`μ` 参数，表示泊松分布的均值，同时也是分布的方差。
        - **seed** (int, 可选) - 随机数种子。必须是一个非负整数，默认 ``None`` 。
        - **dtype** (mindspore.dtype) - 指定数据类型。默认 ``mstype.float32`` 。

    返回：
        Tensor, 形状为 `mindspore.ops.concat([shape, rate.shape], axis=0)` 。
