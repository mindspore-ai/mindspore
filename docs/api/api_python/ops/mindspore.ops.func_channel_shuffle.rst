mindspore.ops.channel_shuffle
=============================

.. py:function:: mindspore.ops.channel_shuffle(x, groups)

    将shape为 :math:`(*, C, H, W)` 的Tensor的通道划分成 :math:`g` 组，并按如下方式重新排列 :math:`(*, \frac{C}{g}, g, H*W)` ，同时保持原始Tensor的shape不变。

    参数：
        - **x** (Tensor) - 被划分输入Tensor。shape为 :math:`(*, C, H, W)` ，数据类型为float16, float32、int8、int16、int32、int64、uint8、uint16、uint32或uint64。
        - **groups** (int) - 通道划分数目。

    返回：
        Tensor，数据类型与 `x` 相同，shape为 :math:`(*, C, H, W)` 。

    异常：
        - **TypeError** - `x` 的数据类型不是float16, float32、int8、int16、int32、int64、uint8、uint16、uint32或uint64。
        - **TypeError** - `x` 的维度小于4。
        - **TypeError** - `groups` 不是正整数。
        - **ValueError** - `x` 的通道数不能被 `groups` 整除。
