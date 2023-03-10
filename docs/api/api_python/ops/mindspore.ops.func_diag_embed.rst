mindspore.ops.diag_embed
=========================

.. py:function:: mindspore.ops.diag_embed(input, offset=0, dim1=-2, dim2=-1)

    生成一个Tensor，其对角线值由 `input` 中的值填充，其余位置置0。如果 `input` 的shape为 :math:`[x_{0}, x_{1}, ..., x_{n-1}, x_{n}]` ，则输出
    shape为将 :math:`x_{n}+|offset|` 插入 :math:`[x_{0}, x_{1}, ..., x_{n-1}]` 的 `dim1` 和 `dim2` 维后得到的向量。

    参数：
        - **input** (Tensor) - 对角线填充值。
        - **offset** (int，可选) - 对角线偏离值。 :math:`offset=0` 为主对角线。

          - 如果 :math:`offset>0` ，填充主对角线上方第 `offset` 条对角线。
          - 如果 :math:`offset<0` ，填充主对角线下方第 `offset` 条对角线。

          默认值：0。

        - **dim1** (int，可选) - 填充对角线的第一个维度。默认值：-2。
        - **dim2** (int，可选) - 填充对角线的第二个维度。默认值：-1。

    返回：
        Tensor，数据类型与 `input` 一致，但输出shape维度比 `input` 高一维。

        - 如果 `keepdims` 为True，则输出shape为： :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)` 。
        - 否则输出shape为： :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不被支持。
        - **TypeError** - `offset` 不是int类型。
        - **TypeError** - `dim1` 或 `dim2` 不是int类型。
        - **ValueError** - `input` 的维度不是1D-6D。
        - **ValueError** - `dim1` 不在[-len(input.shape), len(input.shape))范围内。
        - **ValueError** - `dim2` 不在[-len(input.shape), len(input.shape))范围内。
        - **ValueError** - `dim1` 和 `dim2` 相等。
