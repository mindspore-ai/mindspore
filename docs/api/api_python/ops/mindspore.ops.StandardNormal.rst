mindspore.ops.StandardNormal
============================

.. py:class:: mindspore.ops.StandardNormal(seed=0, seed2=0)

    根据标准正态（高斯）随机数分布生成随机数。

    更多参考详见 :func:`mindspore.ops.standard_normal`。

    参数：
        - **seed** (int) - 随机种子，非负值。默认值：0。
        - **seed2** (int) - 随机种子2，用来防止随机种子冲突，非负值。默认值：0。

    输入：
        - **shape** (tuple) - 目标随机数Tensor的shape。只允许常量值。

    输出：
        Tensor。shape为输入 `shape` 。数据类型支持float32。
