mindspore.ops.flatten
======================

.. py:function:: mindspore.ops.flatten(input, order='C', *, start_dim=1, end_dim=-1)

    沿着从 `start_dim` 到 `end_dim` 的维度，对输入Tensor进行展平。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `order` 将被删除。
        - 参数 `start_dim` 的默认值将从 ``1`` 调整为 ``0`` 。

        变更后的接口签名为 ``mindspore.ops.flatten(input, *, start_dim=0, end_dim=-1)`` 。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **order** (str, 可选) - 仅支持 ``'C'`` 和 ``'F'`` 。 ``'C'`` 表示按行优先顺序 (C风格) 展平， ``'F'`` 表示按列优先顺序 (Fortran风格) 展平。默认 ``'C'`` 。

    关键字参数：
        - **start_dim** (int, 可选) - 要展平的第一个维度。默认 ``1`` 。
        - **end_dim** (int, 可选) - 要展平的最后一个维度。默认 ``-1`` 。

    返回：
        Tensor。如果没有维度被展平，返回原始的输入 `input`，否则返回展平后的Tensor。如果 `input` 是零维Tensor，将会返回一个一维Tensor。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `order` 不是string类型。
        - **ValueError** - `order` 是string类型，但不是 ``'C'`` 或 ``'F'`` 。
        - **TypeError** - `start_dim` 或 `end_dim` 不是int类型。
        - **ValueError** - 规范化后， `start_dim` 大于 `end_dim` 。
        - **ValueError** - `start_dim` 或 `end_dim` 不在 [-input.dim, input.dim-1] 范围内。
