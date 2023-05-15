mindspore.ops.MatrixDiagV3
==========================

.. py:class:: mindspore.ops.MatrixDiagV3(align="RIGHT_LEFT")

    构造以输入Tensor为对角线的矩阵。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.matrix_diag` 。

    参数：
        - **align** (str, 可选) - 可选字符串，指定超对角线和次对角线的对齐方式。
          可选值： ``"RIGHT_LEFT"`` 、 ``"LEFT_RIGHT"`` 、 ``"LEFT_LEFT"`` 、 ``"RIGHT_RIGHT"`` 。
          默认值： ``"RIGHT_LEFT"`` 。

          - ``"RIGHT_LEFT"`` 表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。
          - ``"LEFT_RIGHT"`` 表示将超对角线与左侧对齐（右侧填充行），将次对角线与右侧对齐（左侧填充行）。
          - ``"LEFT_LEFT"`` 表示将超对角线和次对角线均与左侧对齐（右侧填充行）。
          - ``"RIGHT_RIGHT"`` 表示将超对角线与次对角线均右侧对齐（左侧填充行）。

    输入：
        - **x** (Tensor) - 对角线Tensor。
        - **k** (Union[int, Tensor], 可选) - 对角线偏移。int32类型的Tensor。正值表示超对角线，0表示主对角线，负值表示次对角线。k可以是单个整数（对于单个对角线）或一对整数，指定矩阵带的上界和下界，且k[0]不得大于k[1]。该值必须在必须在（-num_rows，num_cols）中。默认值： ``0`` 。
        - **num_rows** (Union[int, Tensor], 可选) - 输出Tensor的行数。int32类型的单值Tensor，若该值为-1，则表示输出Tensor的最内层矩阵是一个方阵，实际行数将由其他输入推导， 即 :math:`num\_rows = x.shape[-1] - min(k[1], 0)` ； 否则，改值必须大于或等于 :math:`x.shape[-1] - min(k[1], 0)` 。默认值： ``-1`` 。
        - **num_cols** (Union[int, Tensor], 可选) - 输出Tensor的列数。int32类型的单值Tensor，若该值为-1，则表示输出Tensor的最内层矩阵是一个方阵，实际列数将由其他输入推导，即 :math:`num\_cols = x.shape[-1] + max(k[0], 0)` ； 否则，改值必须大于或等于 :math:`x.shape[-1] - min(k[1], 0)` 。默认值： ``-1`` 。
        - **padding_value** (Union[int, float, Tensor], 可选) - 填充对角线带外区域的数值，是一个与 `x` 相同的数据类型的单值Tensor。默认值： ``0`` 。

    输出：
        Tensor，与 `x` 的类型相同。
        设 `x` 有r维 :math:`(I, J, ..., M, N)` 。当只给出一条对角线（k是整数或k[0]==k[1]）时，输出Tensor的维度是r + 1，具有shape :math:`(I, J, ..., M, num\_rows, num\_cols)` 。否则，输出Tensor的维度是r，具有shape :math:`(I, J, ..., num\_rows, num\_cols)` 。
