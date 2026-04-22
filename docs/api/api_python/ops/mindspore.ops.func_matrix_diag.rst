mindspore.ops.matrix_diag
=========================

.. py:function:: mindspore.ops.matrix_diag(x, k=0, num_rows=-1, num_cols=-1, padding_value=0, align="RIGHT_LEFT")

    使用 `x` 中的元素值作为输出tensor的对角线，其余用 `padding_value` 填充。

    `num_rows` 和 `num_cols` 为int32类型的单值tensor，若为-1，则表示输出tensor的最内层矩阵是一个方阵。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **k** (Union[int, Tensor], 可选) - 对角线偏移。正值表示超对角线，负值表示次对角线。当k是2个整数，表示子对角线的上界和下界。默认 ``0`` 。
        - **num_rows** (Union[int, Tensor], 可选) - 输出tensor的行数。默认 ``-1`` 。
        - **num_cols** (Union[int, Tensor], 可选) - 输出tensor的列数。默认 ``-1`` 。
        - **padding_value** (Union[int, float, Tensor], 可选) - 填充对角线带外区域的数值。默认 ``0`` 。
        - **align** (str, 可选) - 指定超对角线和次对角线的对齐方式。
          可选 ``"RIGHT_LEFT"`` 、 ``"LEFT_RIGHT"`` 、 ``"LEFT_LEFT"`` 、 ``"RIGHT_RIGHT"`` 。
          默认 ``"RIGHT_LEFT"`` 。

          - ``"RIGHT_LEFT"`` 表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。
          - ``"LEFT_RIGHT"`` 表示将超对角线与左侧对齐（右侧填充行），将次对角线与右侧对齐（左侧填充行）。
          - ``"LEFT_LEFT"`` 表示将超对角线和次对角线均与左侧对齐（右侧填充行）。
          - ``"RIGHT_RIGHT"`` 表示将超对角线和次对角线均与右侧对齐（左侧填充行）。

    返回：
        Tensor
