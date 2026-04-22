mindspore.ops.randperm
========================

.. py:function:: mindspore.ops.randperm(n, seed=0, offset=0, dtype=mstype.int64)

    生成从 0 到 n-1 的整数随机排列。

    .. warning::
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。
        - 2.9.0（不含）之后版本非兼容性变更：`seed` 和 `offset` 将被移除，接口将改为使用 `generator`。

    参数：
        - **n** (Union[Tensor, int]) - 输入上界（不包含）。
        - **seed** (int, 可选) - 随机种子。默认 ``0`` 。当seed为-1时，offset为0，随机数由时间决定。
        - **offset** (int, 可选) - 偏移量。生成随机数，优先级高于随机种子。必须是非负数，默认 ``0`` 。
        - **dtype** (mindspore.dtype, 可选) - 指定数据类型。默认 ``mstype.int64`` 。

    返回：
        Tensor，shape由参数 `n` 决定。
