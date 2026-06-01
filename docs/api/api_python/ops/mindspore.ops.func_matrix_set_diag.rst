mindspore.ops.matrix_set_diag
=============================

.. py:function:: mindspore.ops.matrix_set_diag(x, diagonal, k=0, align="RIGHT_LEFT")

    `mindspore.ops.matrix_set_diag` 从2.9.0版本开始已被弃用，并将在未来版本中被移除。

    返回一个tensor，使用输入 `diagonal` 中的元素值替换 `x` 矩阵的第 `k[0]` 条到第 `k[1]` 条对角线上的元素值。

    参数：
        - **x** (Tensor) - 输入tensor，其秩不小于2。
        - **diagonal** (Tensor) - 输入对角线tensor。
        - **k** (Union[int, Tensor], 可选) - 对角线偏移。正值表示超对角线，负值表示次对角线。当k是2个整数，表示子对角线的上界和下界。默认 ``0`` 。
        - **align** (str, 可选) - 可选字符串，指定超对角线和次对角线的对齐方式。
          可选 ``"RIGHT_LEFT"`` 、 ``"LEFT_RIGHT"`` 、 ``"LEFT_LEFT"`` 、 ``"RIGHT_RIGHT"`` 。
          默认 ``"RIGHT_LEFT"`` 。

          - ``"RIGHT_LEFT"`` 表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。
          - ``"LEFT_RIGHT"`` 表示将超对角线与左侧对齐（右侧填充行），将次对角线与右侧对齐（左侧填充行）。
          - ``"LEFT_LEFT"`` 表示将超对角线和次对角线均与左侧对齐（右侧填充行）。
          - ``"RIGHT_RIGHT"`` 表示将超对角线和次对角线均与右侧对齐（左侧填充行）。

    返回：
        Tensor
