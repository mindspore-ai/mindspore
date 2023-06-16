mindspore.ops.MaskedSelect
===========================

.. py:class:: mindspore.ops.MaskedSelect

    返回一个一维张量，其中的内容是 `x` 张量中对应于 `mask` 张量中True位置的值。`mask` 的shape与 `x` 的shape不需要一样，但必须符合广播规则。

    输入：
        - **x** (Tensor) - 任意维度输入Tensor。
        - **mask** (Tensor[bool]) - 掩码Tensor，数据类型为bool，shape与 `x` 一致。

    输出：
        一维Tensor，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 或 `mask` 不是Tensor。
        - **TypeError** - `mask` 不是bool类型的Tensor。
