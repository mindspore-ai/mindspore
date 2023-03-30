mindspore.ops.RandomShuffle
============================

.. py:class:: mindspore.ops.RandomShuffle(seed=0, seed2=0)

    随机沿着Tensor的第一维度进行随机打乱操作。

    参数：
        - **seed** (int，可选) - 随机数种子。如果 `seed` 或者 `seed2` 被设置为非零，则使用这个非零值。否则使用一个随机生成的种子。 `seed` 必须为非负数。默认值：0。
        - **seed2** (int，可选) - 为了避免种子碰撞的第二个种子。如果 `seed` 为0，则 `seed2` 将被用作随机生成器的种子。 `seed2` 必须是非负数。默认值：0。

    输入：
        - **x** (tuple) - 需要进行打乱操作的Tensor。

    输出：
        Tensor。输出的shape和类型与输入 `x` 相同。

    异常：
        - **TypeError** - 如果 `seed` 或 `seed2` 的数据类型不是int。
