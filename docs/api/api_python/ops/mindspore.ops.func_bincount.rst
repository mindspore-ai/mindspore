mindspore.ops.bincount
======================

.. py:function:: mindspore.ops.bincount(x, weights=None, minlength=0)

    计算非负整数数组中每个值的出现次数。（大小为1的）bins的数量为 `x` 中的最大值加1。
    如果指定了 `minlength`，则输出数组中至少会有此数量的bins（如果需要，它会更长，具体取决于 `x` 的内容）。
    每个bin给出其索引值在 `x` 中的出现次数。如果指定了 `weights`，则输入数组由其加权，即如果在位置 `i` 处的值 `n`，则 `out[n]+=weight[i]` 而不是 `out[n]+=1`。

    参数：
        - **x** (Tensor) - 一维的Tensor。
        - **weights** (Tensor, 可选) - 权重，与 `x` shape相同的tensor。默认值：None。
        - **minlength** (int, 可选) - 输出Tensor的最小bin的数量。默认值：0。
    
    返回：
        Tensor，如果输入为非空，输出shape为[max(input)+1]的Tensor，否则shape为[0]。

    异常：
        - **TypeError** - 如果 `x` 或 `weights` 不是Tensor。
        - **ValueError** - 如果 `x` 不是一维的，或者 `x` 和 `weights` 不具有相同的shape。
        - **ValueError** - 如果 `minlength` 是负整数。
