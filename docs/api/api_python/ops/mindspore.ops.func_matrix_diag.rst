mindspore.ops.matrix_diag
=========================

.. py:function:: mindspore.ops.matrix_diag(x, k=0, num_rows=-1, num_cols=-1, padding_value=0, align="RIGHT_LEFT")

    返回一个Tensor，其k[0]到k[1]的对角线特定为给定对角线Tensor，其余值均填充为 `padding_value` 。
    通过 `num_rows` 和 `num_cols` 指定输出最内层矩阵的维度，其维度大小需要符合要求。如果两者都没有指定，那么算子假定输出Tensor最内层的矩阵是方阵，并从输入 `k` 和输入 `x` 最内层的维度推断出输出的具体维度大小。如果 `num_rows` 和 `num_cols` 仅指定其中一个，那么算子将推导出最小的合法值作为输出的维度。
    此外，当只有一条对角线时（即当k为整数或者k[0]==k[1]），`x` 的第一维到倒数第二维都属于批量的范围。否则倒数第二维不属于批量的维度。

    参数：
        - **x** (Tensor) - 对角线Tensor。
        - **k** (Union[int, Tensor], 可选) - int32类型的Tensor。对角线偏移。正值表示超对角线，0表示主对角线，负值表示次对角线。k可以是单个整数（对于单个对角线）或一对整数，指定矩阵带的上界和下界，且k[0]不得大于k[1]。该值必须在必须在（-num_rows，num_cols）中。默认值：0。
        - **num_rows** (Union[int, Tensor], 可选) - int32类型的单值Tensor，表示输出Tensor的行数。若该值为-1，则表示输出Tensor的最内层矩阵是一个方阵，实际行数将由其他输入推导。默认值：-1。
        - **num_cols** (Union[int, Tensor], 可选) - int32类型的单值Tensor，表示输出Tensor的列数。若该值为-1，则表示输出Tensor的最内层矩阵是一个方阵，实际列数将由其他输入推导。默认值：-1。
        - **padding_value** (Union[int, float, Tensor], 可选) - 与 `x` 相同的数据类型的单值Tensor，表示填充对角线带外区域的数值，默认值：0。
        - **align** (str, 可选) - 一个字符串，指定超对角线和次对角线的对齐方式。可选字符串有：RIGHT_LEFT、"LEFT_RIGHT"、"LEFT_LEFT"、"RIGHT_RIGHT"。例如，"RIGHT_LEFT"表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。默认值："RIGHT_LEFT"。

    返回：
        Tensor，与 `x` 的类型相同。
        设 `x` 有r维 `(I， J， ...， M， N)` 。当只给出一条对角线（k是整数或k[0]==k[1]）时，输出Tensor的维度是r + 1，具有shape `(I，J，…，M，num_rows，num_cols)` 。否则，输出Tensor的维度是r，具有shape `(I，J，…，num_rows，num_cols)` 。

    异常：
        - **TypeError** - `x` 不为Tensor。
        - **TypeError** - `x` 与 `padding_value` 数据类型不同。
        - **TypeError** - `k` 、 `num_rows` 、 `num_cols` 数据类型不为int32。
        - **ValueError** - `k` 的维度不为0或1。
        - **ValueError** - `padding_value` 、 `num_rows` 、 `num_cols` 的维度不为0。
        - **ValueError** - `k` 的大小不为1或2。
        - **ValueError** - `k` 的取值不在 (-num_rows, num_cols) 范围内。
        - **ValueError** - 当k[0] != k[1]时，k[1]小于k[0]。
        - **ValueError** - 当k为整数或k[0] == k[1]时， `x` 的维度小于1。
        - **ValueError** - 当k[0] != k[1]时，`x` 的维度小于2。
        - **ValueError** - 当k[0] != k[1]时，x.shape[-2]不等于k[1] - k[0] + 1。
        - **ValueError** - `num_rows` 和 `num_cols` 与 `x` 的维度和 `k` 的值不匹配。
        - **ValueError** - `align` 取值不在合法值集合内。
