mindspore.ops.matrix_set_diag
=============================

.. py:function:: mindspore.ops.matrix_set_diag(x, diagonal, k=0, align="RIGHT_LEFT")

    返回具有新的对角线值的批处理矩阵张量。
    给定输入 `x` 和对角线 `diagonal` ，此操作返回与 `x` 具有相同形状和值的张量，但返回的张量除开最内层矩阵的对角线。这些值将被对角线中的值覆盖。如果某些对角线比 `max_diag_len` 短，则需要被填充。
    其中 `max_diag_len` 指的是对角线的最长长度。
    `diagonal` 的维度 :math:`shape[-2]` 必须等于对角线个数 `num_diags` :math:`k[1] - k[0] + 1`， `diagonal` 的维度 :math:`shape[-1]` 必须
    等于最长对角线值 `max_diag_len`  :math:`min(x.shape[-2] + min(k[1], 0), x.shape[-1] + min(-k[0], 0))` 。
    设 `x` 具有 `r + 1` 维 :math:`[I, J, ..., L, M, N]` 。当 `k` 是整数或 :math:`k[0] == k[1]` 时，对角线 `diagonal` 具有形状
    为 :math:`[I, J, ..., L, max\_diag\_len]` 。否则，它具有形状为 :math:`[I, J, ... L, num\_diags, max\_diag\_len]` 。

    参数：
        - **x** (Tensor) - 输入Tensor，其维度为 `r+1` 需要满足 `r >=1` 。
        - **diagonal** (Tensor) - 输入对角线Tensor，具有与 `x` 相同的数据类型。当 `k` 是整数或 :math:`k[0] == k[1]` 时，其为维度 `r` ，否则，其维度 `r + 1` 。
        - **k** (Union[int, Tensor], 可选) - int32常量或int32类型Tensor。对角线偏移。正值表示超对角线，0表示主对角线，负值表示次对角线。k可以是单个整数（对于单个对角线）或一对整数，指定矩阵带的上界和下界，且 `k[0]` 不得大于 `k[1]` 。该值必须在必须在 :math:`(-x.shape[-2], x.shape[-1])` 中。默认值：0。
        - **align** (str, 可选) - 字符串，指定超对角线和次对角线的对齐方式。可选值："RIGHT_LEFT"、"LEFT_RIGHT"、"LEFT_LEFT"、"RIGHT_RIGHT"。例如，"RIGHT_LEFT"表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。默认值："RIGHT_LEFT"。

    返回：
        Tensor，与 `x` 的类型相同。

        设 `x` 有 `r+1` 维 :math:`[I， J， ...， M， N]` 。输出Tensor的维度为 `r+1` ，为 :math:`[I, J, ..., L, M, N]` 。

    异常：
        - **TypeError** - `x` 或者 `diagonal` 不为Tensor。
        - **TypeError** - `x` 与 `diagonal` 数据类型不同。
        - **TypeError** - `k` 的数据类型不为int32。
        - **ValueError** - `align` 取值不在合法值集合内。
        - **ValueError** - `k` 的维度不为0或1。
        - **ValueError** - `x` 的维度不大于等于2。
        - **ValueError** - `k` 的大小不为1或2。
        - **ValueError** - 当 `k` 的大小为2时，k[1]小于k[0]。
        - **ValueError** - 对角线 `diagonal` 的维度与输入 `x` 的维度不匹配。
        - **ValueError** - 对角线 `diagonal` 的维度信息与输入 `x` 的维度信息不匹配。
        - **ValueError** - 对角线 `diagonal` 的维度 :math:`shape[-2]` 不等于与对角线个数 `num_diags` :math:`k[1]-k[0]+1` 。
        - **ValueError** - `k` 的取值不在 :math:`(-x.shape[-2], x.shape[-1])` 范围内。
        - **ValueError** - 对角线 `diagonal` 的维度shape[-1] 不等于最长对角线长度 `max_diag_len` :math:`min(x.shape[-2] + min(k[1], 0), x.shape[-1] + min(-k[0], 0))` 。

