mindspore.ops.MatrixSetDiagV3
=============================

.. py:class:: mindspore.ops.MatrixSetDiagV3(align="RIGHT_LEFT")

    返回一批具有新的对角线值的矩阵的Tensor。
    给定输入 `x` 和对角线 `diagonal` ，此操作返回与 `x` 具有相同shape和值的Tensor。但返回的Tensor除了最内层矩阵的对角线，
    这些值将被对角线中的值覆盖。
    
    如果某些对角线比 `max_diag_len` 短，则需要被填充，其中 `max_diag_len` 指对角线的最长长度。
    `diagonal` 的维度 :math:`shape[-2]` 必须等于对角线个数 `num_diags` ， :math:`num\_diags = k[1] - k[0] + 1`，
    `diagonal` 的维度 :math:`shape[-1]` 必须等于最长对角线值 `max_diag_len` ，
    :math:`max\_diag\_len = min(x.shape[-2] + min(k[1], 0), x.shape[-1] + min(-k[0], 0))` 。

    设 `x` 是一个n维Tensor，shape为： :math:`(d_1, d_2, ..., d_{n-2}, d_{n-1}, d_n)` 。
    当 `k` 是一个整数或 :math:`k[0] == k[1]` 时， `diagonal` 为n-1维Tensor，shape为 :math:`(d_1, d_2, ..., d_{n-2}, max\_diag\_len)` 。
    否则， `diagonal` 与 `x` 维度一致，其shape为 :math:`(d_1, d_2, ..., d_{n-2}, num\_diags, max\_diag\_len)` 。

    参数：
        - **align** (str，可选) - 可选字符串，指定超对角线和次对角线的对齐方式。
          可选值："RIGHT_LEFT"、"LEFT_RIGHT"、"LEFT_LEFT"、"RIGHT_RIGHT"。
          默认值："RIGHT_LEFT"。

          - "RIGHT_LEFT"表示将超对角线与右侧对齐（左侧填充行），将次对角线与左侧对齐（右侧填充行）。
          - "LEFT_RIGHT"表示将超对角线与左侧对齐（右侧填充行），将次对角线与右侧对齐（左侧填充行）。
          - "LEFT_LEFT"表示将超对角线和次对角线均与左侧对齐（右侧填充行）。
          - "RIGHT_RIGHT"表示将超对角线与次对角线均右侧对齐（左侧填充行）。

    输入：
        - **x** (Tensor) - n维Tensor，其中 :math:`n >= 2` 。
        - **diagonal** (Tensor) - 输入对角线Tensor，具有与 `x` 相同的数据类型。
          当 `k` 是整数或 :math:`k[0] == k[1]` 时，其为维度 `n-1` ，否则，其维度为 `n` 。
        - **k** (Tensor) - int32类型的Tensor。对角线偏移量。正值表示超对角线，0表示主对角线，负值表示次对角线。
          `k` 可以是单个整数（对于单个对角线）或一对整数，分别指定矩阵带的上界和下界，且 `k[0]` 不得大于 `k[1]` 。
          其值必须在 :math:`(-x.shape[-2], x.shape[-1])` 中。采用图模式时，输入 `k` 必须是常量Tensor。

    输出：
        Tensor，数据类型和shape与 `x` 相同。


    异常：
        - **TypeError** - 若任一输入不是Tensor。
        - **TypeError** - `x` 与 `diagonal` 数据类型不同。
        - **TypeError** - `k` 的数据类型不为int32。
        - **ValueError** - `align` 取值不在合法值集合内。
        - **ValueError** - `k` 的维度不为0或1。
        - **ValueError** - `x` 的维度不大于等于2。
        - **ValueError** - `k` 的大小不为1或2。
        - **ValueError** - 当 `k` 的大小为2时， `k[1]` 小于 `k[0]` 。
        - **ValueError** - 对角线 `diagonal` 的维度与输入 `x` 的维度不匹配。
        - **ValueError** - 对角线 `diagonal` 的shape与输入 `x` 不匹配。
        - **ValueError** - 对角线 `diagonal` 的维度 :math:`shape[-2]` 不等于与对角线个数 `num_diags` ，
          :math:`num\_diags = k[1]-k[0]+1` 。
        - **ValueError** - `k` 的取值不在 :math:`(-x.shape[-2], x.shape[-1])` 范围内。
        - **ValueError** - 对角线 `diagonal` 的维度 :math:`shape[-1]` 不等于最长对角线长度 `max_diag_len`，
          :math:`max\_diag\_len = min(x.shape[-2] + min(k[1], 0), x.shape[-1] + min(-k[0], 0))` 。
