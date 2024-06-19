mindspore.Symbol
================

.. py:class:: mindspore.Symbol(max=0, min=1, divisor=1, remainder=0, unique=False, **kawgs)

    符号，用来传递张量形状的符号信息（symbolic shape）的数据结构。

    对于动态shape网络，相比只设置 `shape` 的未知维度（ ``None`` ），提供未知维度的数学符号信息能帮助框架更好地优化计算图，提高网络执行性能。

    参数：
        - **max** (int) - 最大值，表示此维度的最大长度。只有当它大于 `min` 值时才有效。默认值： ``0`` 。
        - **min** (int) - 最小值，表示此维度的最小长度，要求是正数。默认值： ``1`` 。
        - **divisor** (int) - 约数 :math:`d` 。默认值： ``1`` 。
        - **remainder** (int) - 余数 :math:`r`。与 `divisor` 一起表示符号值为 :math:`d * N + r, N \ge 1` 。 默认值： ``0`` 。
        - **unique** (bool) - 符号具有唯一值。当这个 `Symbol` 对象被 `Tensor` 多次引用时，如果 `unique` 为 ``True`` ，表示用到这个 `Symbol` 对象的维度的长度都相等；否则表示只共享符号信息，不一定相等。默认值： ``False`` 。

    输出：
        Symbol。

    异常：
        - **TypeError** - 如果 `max` 、 `min` 、 `divisor` 、 `remainder` 不是整数。
        - **TypeError** - 如果 `unique` 不是布尔值。
        - **ValueError** - 如果 `min` 不是正数。
        - **ValueError** - 如果 `divisor` 不是正数。
        - **ValueError** - 如果 `remainder` 不在区间 :math:`[0, d)` 内。
