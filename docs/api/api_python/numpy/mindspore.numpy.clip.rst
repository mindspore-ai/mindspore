mindspore.numpy.clip
====================

.. py:function:: mindspore.numpy.clip(x, xmin, xmax, dtype=None)

    裁剪(限制)数组的值。给定一个区间，区间外的值会被限制到区间边界。例如，如果给定区间 :math:`[0,1]` ，则小于0的值变为0，大于1的值变为1。

    参数：
        - **x** (Tensor) - 包含需要限制的元素的Tensor。
        - **xmin** (Tensor, scalar, None) - 最小值。如果为 ``None``，则不在下边界进行限制。 `xmin` 和 `xmax` 不可以同时为None。
        - **xmax** (Tensor, scalar, None) - 最大值。如果为 ``None``，则不在上边界进行限制。 `xmin` 和 `xmax` 不可以同时为None。如果 `xmin` 或 `xmax` 是Tensor，则这三个Tensor将会广播以匹配它们的shape。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor，一个包含x的元素的Tensor，但小于 `xmin` 的值替换为 `xmin` ，大于 `xmax` 的值替换为 `xmax` 。

    异常：
        - **TypeError** - 如果输入的类型不符合上述规定。
        - **ValueError** - 如果 `x` 、 `xmin` 和 `xmax` 的shape无法被广播，或者 `xmin` 和 `xmax` 同时为 `None` 。
