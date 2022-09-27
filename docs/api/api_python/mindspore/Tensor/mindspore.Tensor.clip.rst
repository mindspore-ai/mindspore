mindspore.Tensor.clip
=====================

.. py:method:: mindspore.Tensor.clip(xmin, xmax, dtype=None)

    裁剪Tensor中的值。

    给定一个区间，区间外的值将被裁剪到区间边缘。
    例如，如果指定的间隔为 :math:`[0, 1]` ，则小于0的值将变为0，大于1的值将变为1。

    .. note::
        目前不支持裁剪 `xmin=nan` 或 `xmax=nan` 。

    参数：
        - **xmin** (Tensor, scalar, None) - 最小值。如果值为None，则不在间隔的下边缘执行裁剪操作。`xmin` 或 `xmax` 只能有一个为None。
        - **xmax** (Tensor, scalar, None) - 最大值。如果值为None，则不在间隔的上边缘执行裁剪操作。`xmin` 或 `xmax` 只能有一个为None。如果 `xmin` 或 `xmax` 是Tensor，则三个Tensor将被广播进行shape匹配。
        - **dtype** (mindspore.dtype, 可选) - 覆盖输出Tensor的dtype。默认值为None。

    返回：
        Tensor，含有输入Tensor的元素，其中values < `xmin` 被替换为 `xmin` ，values > `xmax` 被替换为 `xmax` 。

    异常：
        - **TypeError** - 输入的类型与Tensor不一致。
        - **ValueError** - 输入与Tensor的shape不能广播，或者 `xmin` 和 `xmax` 都是 `None` 。