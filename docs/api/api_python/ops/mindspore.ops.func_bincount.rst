mindspore.ops.bincount
======================

.. py:function:: mindspore.ops.bincount(input, weights=None, minlength=0)

    统计一个输入tensor中每个非负整数出现的次数。

    - 如果不指定 `minlength` ，输出tensor的长度为max( `input` )+1。
    - 如果指定 `minlength`，则输出tensor的长度为max([max( `input` )+1, `minlength`])。

    如果指定了 `weights`，对输出的结果进行加权处理，如果 `n` 是输入tensor中索引 `i` 处的值，即 `out[n]+=weight[i]` 而不是 `out[n]+=1`。

    .. note::
        如果 `input` 含有负数元素，则结果未定义。

    参数：
        - **input** (Tensor) - 一维输入tensor。
        - **weights** (Tensor, 可选) - 权重。默认 ``None`` 。
        - **minlength** (int, 可选) - 输出tensor的最小长度。默认 ``0`` 。

    返回：
        Tensor，如果输入为非空，输出shape为[max(input)+1]的tensor，否则shape为[0]。
