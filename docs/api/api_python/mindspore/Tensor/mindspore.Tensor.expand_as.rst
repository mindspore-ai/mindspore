mindspore.Tensor.expand_as
==========================

.. py:method:: mindspore.Tensor.expand_as(other) -> Tensor

    将输入张量的shape扩展为另一个输入张量的shape。
    输出张量的维度必须遵守广播规则，即输入张量的shape维度小于或者等于另一个输入张量的shape维度。

    参数：
        - **other** (Tensor) - 目标张量。其shape为输入张量扩展的目标shape。

    返回：
        维度与另一个输入张量 `other` 相同的Tensor，且数据类型与输入张量 `self` 相同。

    异常：
        - **TypeError** - 如果参数 `other` 不是张量。
        - **ValueError** - 如果 `self` 和 `other` 的shape不兼容。

    .. py:method:: mindspore.Tensor.expand_as(x) -> Tensor
        :noindex:

    将输入张量的维度扩展为目标张量的维度。

    参数：
        - **x** (Tensor) - 目标张量。其shape必须符合扩展的规则。

    返回：
        维度与目标张量的相同的Tensor。
