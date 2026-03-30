mindspore.nn.ChannelShuffle
============================

.. py:class:: mindspore.nn.ChannelShuffle(groups)

    将shape为 :math:`(*, C, H, W)` 的Tensor的通道划分成 :math:`g` 组，并按如下方式重新排列 :math:`(*, \frac{C}{g}, g, H*W)` ，同时在最终输出中保持原始Tensor的shape。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：前向输入 `x` 将改为 `input`。

    参数：
        - **groups** (int) - 划分通道的组数，必须大于0。在上述公式中表示为 :math:`g` 。

    输入：
        - **x** (Tensor) - Tensor的shape :math:`(*, C_{in}, H_{in}, W_{in})` 。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `groups` 非正整数。
        - **ValueError** - `groups` 小于1。
        - **ValueError** - `x` 的维度小于3。
        - **ValueError** - `x` 的通道数不能被 `groups` 整除。
