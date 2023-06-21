mindspore.ops.MaskedScatter
===========================

.. py:class:: mindspore.ops.MaskedScatter

    返回一个Tensor。根据 `mask` 和 `updates` 更新输入Tensor的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x** (Tensor) - 被更新输入Tensor。
        - **mask** (Tensor[bool]) - 指示应修改或替换哪些元素的掩码Tensor， `mask` 和 `x` 的shape必须相等或者两者的shape可以广播。
        - **updates** (Tensor) - 要散播到目标张量或数组中的值。其数据类型与 `x` 相同。 `updates` 中的元素数量必须大于等于 `mask` 中的True元素的数量。

    输出：
        Tensor，其数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 、 `mask` 或者 `updates` 不是Tensor。
        - **TypeError** - 如果 `x` 的数据类型不被支持。
        - **TypeError** - 如果 `mask` 的dtype不是bool。
        - **TypeError** - 如果 `x` 的维度数小于 `mask` 的维度数。
        - **ValueError** - 如果 `mask` 不能广播到 `x` 。
        - **ValueError** - 如果 `updates` 中的元素数目小于 `mask` 中的True元素的数量。
