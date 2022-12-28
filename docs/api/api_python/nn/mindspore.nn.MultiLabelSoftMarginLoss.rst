mindspore.nn.MultiLabelSoftMarginLoss
======================================

.. py:class:: mindspore.nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')

    基于最大熵计算用于多标签优化的损失。计算公式如下。

    .. math::
        \mathcal{L}_{D} = - \frac{1}{|D|}\sum_{i = 0}^{|D|}\left(
        y_{i}\ln\frac{1}{1 + e^{- x_{i}}} + \left( 1 - y_{i}
        \right)\ln\frac{1}{1 + e^{x_{i}}} \right)

    :math:`\mathcal{L}_{D}` 为损失值，:math:`y_{i}` 为 `target` ,
    :math:`x_{i}` 为 `x` 。如果 `weight` 不为None，将会和每个分类的loss相乘。

    参数：
        - **weight** (Union[Tensor, int, float]) - 每个类别的缩放权重。默认值：None。
        - **reduction** (str) - 指定应用于输出结果的计算方式。取值为"mean"，"sum"，或"none"。默认值："mean"。

    输入：
        - **x** (Tensor) - shape为(N, C)的Tensor，N为batch size，C为类别个数。
        - **target** (Tensor) - 目标值，数据类型和shape与 `x` 的相同。

    输出：
        Tensor，数据类型和 `x` 相同。如果 `reduction` 为"none"，其shape为(N)。否则，其shape为0。

    异常：
        - **TypeError** - `x` 或 `target` 的维度不等于2。
