mindspore.ops.ChannelShuffle
============================

.. py:class:: mindspore.ops.ChannelShuffle(group)

    将shape为 :math:`(*, C, H, W)` 的Tensor的通道划分成 :math:`g` 组，并按如下方式重新排列 :math:`(*, \frac C g, g, H*W)` ，
    同时保持原始Tensor的shape不变。

    .. warning::
        这是一个实验性API，后续可能修改或删除。
    
    更多详情请查看： :class:`mindspore.ops.channel_shuffle` 。
