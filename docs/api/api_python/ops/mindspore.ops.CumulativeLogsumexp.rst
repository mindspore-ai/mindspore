mindspore.ops.CumulativeLogsumexp
==================================

.. py:class:: mindspore.ops.CumulativeLogsumexp(exclusive=False, reverse=False)

    计算输入 `x` 沿轴 `axis` 的累积LogSumExp函数值。即：在参数均为默认值的情况下，若输入 `x` 为[a, b, c]，则输出为[a, log(exp(a) + exp(b)), log(exp(a) + exp(b) + exp(c))]。

    参数：
        - **exclusive** (bool, 可选) - 如果为 ``True`` ，将在计算时跳过最后一个元素，此时输出为：[-inf, a, log(exp(a) * exp(b))]，其中-inf在输出时出于性能原因将以一个极小负数的形式呈现。默认值： ``False`` 。
        - **reverse** (bool, 可选) - 如果为 ``True`` ，将对 `x` 指定轴的元素进行翻转后再计算函数累积值，同时再将该计算结果进行翻转，此时输出为：[log(exp(c) + exp(b) + exp(a)), log(exp(c) + exp(b)), c]。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 数据类型需为float16、float32或float64之一，维度必须大于0。
        - **axis** (Tensor) - 0D Tensor，表示 `x` 中将进行运算的维度，数据类型需为int16、int32或int64之一，取值范围应在区间[-rank(x), rank(x))中。默认值： ``0`` 。

    输出：
        与 `x` 的shape和数据类型一致的Tensor。

    异常：
        - **TypeError** - 输入 `x` 或 `axis` 不是Tensor。
        - **TypeError** -  `x` 的数据类型不是float16、float32或float64之一。
        - **TypeError** -  `axis` 的数据类型不是int16、int32或int64之一。
        - **TypeError** - 参数 `exclusive` 或 `reverse` 不是布尔值。
        - **ValueError** -  `x` 的维度不大于0。
        - **RuntimeError** -  `axis` 中的值不在[-rank(x), rank(x))中。