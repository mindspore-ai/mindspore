mindspore.ops.matrix_diag_part
==============================

.. py:function:: mindspore.ops.matrix_diag_part(x, k=0, padding_value=0, align="RIGHT_LEFT")

    返回输入Tensor的对角线部分。
    返回输入Tensor，内容为输入 `x` 的第k[0]到k[1]个对角线中的值。有些对角线的长度小于 `max_diag_len`，此时会使用 `padding_value` 填充。在图模式中，输入 `k` 和 `padding_value` 必须为常量Tensor。

    参数：
        - **x** (Tensor) - 输入Tensor，维度r需要满足 r >= 2。
        - **k** (Union[int, Tensor], optional) - int或int32类型的Tensor。对角线偏移。正值表示超对角线，0表示主对角线，负值表示次对角线。k可以是单个整数（对于单个对角线）或一对整数，指定矩阵带的上界和下界，且k[0]不得大于k[1]。该值必须在必须在（-x.shape[-2], x.shape[-1]）中。默认值：0。
        - **padding_value** (Union[int, float, Tensor], optional) - 与 `x` 相同的数据类型的单值Tensor，表示填充对角线带外区域的数值，默认值：0。
        - **align** (str, optional) - 一个字符串，指定超对角线和次对角线的对齐方式。可选字符串有："RIGHT_LEFT"、"LEFT_RIGHT"、"LEFT_LEFT"、"RIGHT_RIGHT"。例如，"RIGHT_LEFT"表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。默认值："RIGHT_LEFT"。

    返回：
        Tensor，与 `x` 的类型相同。

        设 `x` 有r维 `(I， J， ...， M， N)` 。设 `max_diag_len` 为所有对角线长度中的最大值，则 :math:`max\_diag\_len = min(M + min(k[1], 0), N + min(-k[0], 0))`。设 `num_diags` 为输出的维度数，则有 :math:`num\_diags = k[1] - k[0] + 1`。如果 :math:`num\_diags == 1`，则输出Tensor的维度为r - 1，分别为 :math:`[I, J, ..., L, max\_diag\_len]`。否则，输出Tensor的维度为r，分别为 :math:`[I, J, ..., L, num\_diags, max\_diag\_len]`。

    异常：
        - **TypeError** - `x` 不为Tensor。
        - **TypeError** - `x` 与 `padding_value` 数据类型不同。
        - **TypeError** - `k` 的数据类型不为int32。
        - **ValueError** - `align` 取值不在合法值集合内。
        - **ValueError** - `k` 的维度不为0或1。
        - **ValueError** - `padding_value` 的维度不为0。
        - **ValueError** - `x` 的维度不大于等于2。
        - **ValueError** - `k` 的大小不为1或2。
        - **ValueError** - 当 `k` 的大小为2时，k[1]小于k[0]。
        - **ValueError** - `k` 的取值不在 (-x.shape[-2], x.shape[-1]) 范围内。
