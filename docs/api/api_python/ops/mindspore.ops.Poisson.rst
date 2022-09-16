mindspore.ops.Poisson
=====================

.. py:class:: mindspore.ops.Poisson(seed=0, seed2=0)

    生成 Poisson 分布的随机数。

    .. math::

        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    参数：
        - **seed** (int) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值：0。
        - **seed2** (int) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值：0。

    输入：
        - **shape** (tuple) - 待生成的随机 Tensor 的 shape。只支持常量值。
        - **mean** (Tensor) - Poisson 分布的期望，也就是上面公式中的 μ。其值必须大于0。数据类型为 float32。

    输出：
        Tensor。shape是输入 `shape` 和 `mean` 广播后的 shape。数据类型为 int32。

    异常：
        - **TypeError** - `seed` 或 `seed2` 的数据类型不是 int。
        - **TypeError** - `shape` 不是 tuple。
        - **TypeError** - `mean` 不是数据类型为 float32 的 Tensor。
