mindspore.ops.MatrixDiagPartV3
==============================

.. py:class:: mindspore.ops.MatrixDiagPartV3(align="RIGHT_LEFT")

    返回Tensor的对角线部分。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.matrix_diag_part`。

    参数：
        - **align** (str, 可选) - 可选字符串，指定超对角线和次对角线的对齐方式。
          可选值： ``"RIGHT_LEFT"`` 、 ``"LEFT_RIGHT"`` 、 ``"LEFT_LEFT"`` 、 ``"RIGHT_RIGHT"`` 。
          默认值： ``"RIGHT_LEFT"`` 。

          - ``"RIGHT_LEFT"`` 表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。
          - ``"LEFT_RIGHT"`` 表示将超对角线与左侧对齐（右侧填充行），将次对角线与右侧对齐（左侧填充行）。
          - ``"LEFT_LEFT"`` 表示将超对角线和次对角线均与左侧对齐（右侧填充行）。
          - ``"RIGHT_RIGHT"`` 表示将超对角线与次对角线均右侧对齐（左侧填充行）。

    输入：
        - **x** (Tensor) - 输入Tensor，维度r需要满足 r >= 2。
        - **k** (Tensor) - int或int32类型的Tensor。对角线偏移。正值表示超对角线，0表示主对角线，负值表示次对角线。k可以是单个整数（对于单个对角线）或一对整数，指定矩阵带的上界和下界，且k[0]不得大于k[1]。该值必须在必须在（-x.shape[-2], x.shape[-1]）中。
        - **padding_value** (Tensor) - 与 `x` 相同的数据类型的单值Tensor，表示填充对角线带外区域的数值 。

    输出：
        Tensor，与 `x` 的类型相同。

        - 设 `x` 有r维 :math:`(I, J, ..., M, N)` 。设 `max_diag_len` 为所有对角线长度中的最大值，则 :math:`max\_diag\_len = min(M + min(k[1], 0), N + min(-k[0], 0))`。
        - 设 `num_diags` 为输出的维度数，则有 :math:`num\_diags = k[1] - k[0] + 1`。如果 :math:`num\_diags == 1`，则输出Tensor的维度为r - 1，分别为 :math:`(I, J, ..., L, max\_diag\_len)`。否则，输出Tensor的维度为r，分别为 :math:`(I, J, ..., L, num\_diags, max\_diag\_len)` 。
