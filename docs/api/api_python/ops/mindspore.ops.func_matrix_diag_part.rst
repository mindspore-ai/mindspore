mindspore.ops.matrix_diag_part
==============================

.. py:function:: mindspore.ops.matrix_diag_part(x, k, padding_value, align="RIGHT_LEFT")

    返回一个tensor，包含输入tensor `x` 的第 `k[0]` 到 `k[1]` 个对角线中的值，缺失部分使用 `padding_value` 填充。

    在graph mode中，输入 `k` 和 `padding_value` 必须为常量tensor。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (Tensor) - 输入tensor，其秩不小于2。
        - **k** (Union[int, Tensor], 可选) - 对角线偏移。正值表示超对角线，负值表示次对角线。当k是2个整数，表示子对角线的上界和下界。
        - **padding_value** (Tensor) - 填充对角线带外区域的数值。
        - **align** (str, 可选) - 指定超对角线和次对角线的对齐方式。
          可选 ``"RIGHT_LEFT"`` 、 ``"LEFT_RIGHT"`` 、 ``"LEFT_LEFT"`` 、 ``"RIGHT_RIGHT"`` 。
          默认 ``"RIGHT_LEFT"`` 。

          - ``"RIGHT_LEFT"`` 表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。
          - ``"LEFT_RIGHT"`` 表示将超对角线与左侧对齐（右侧填充行），将次对角线与右侧对齐（左侧填充行）。
          - ``"LEFT_LEFT"`` 表示将超对角线和次对角线均与左侧对齐（右侧填充行）。
          - ``"RIGHT_RIGHT"`` 表示将超对角线和次对角线均与右侧对齐（左侧填充行）。

    返回：
        Tensor
