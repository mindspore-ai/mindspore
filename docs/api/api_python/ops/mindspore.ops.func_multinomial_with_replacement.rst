mindspore.ops.multinomial_with_replacement
==========================================

.. py:function:: mindspore.ops.multinomial_with_replacement(x, seed, offset, numsamples, replacement=False)

    返回一个Tensor，其中每行包含从重复采样的多项式分布中抽取的 `numsamples` 个索引。与 `Multinomial` 不同， `MultinomialWithReplacement` 允许多次选择相同的结果。

    .. note::
        输入的行不需要求和为1（在这种情况下，使用值作为权重），但必须是非负的、有限的，并且具有非零和。

    参数：
        - **x** (Tensor) - 包含概率的累积和的输入Tensor，必须为一维或二维。
        - **seed** (int) - 如果将随机种子设置为-1，并将 `offset` 设置为0，则随机数生成器将使用随机种子进行种植。否则，将使用给定的随机数种子。支持的dtype：int64。
        - **offset** (int) - 为避免种子冲突设置的偏移量。支持的dtype：int64。
        - **numsamples** (int) - 抽取样本量，必须大于零。
        - **replacement** (bool，可选) - 是否有放回地抽取。默认值： ``False`` 。

    返回：
        Tensor，具有与输入 `x` 有相同的行。每行的采样索引数为 `numsamples` 。

    异常：
        - **TypeError** - 如果 `x` 不是1D或2DTensor。
        - **TypeError** - 如果 `x` 数据类型不是float16、float32或float64。
        - **TypeError** - 如果 `num_sample` 不是int类型。
        - **TypeError** - 如果 `replacement` bool类型。
        - **ValueError** - 如果 `replacement` 为False的时候， `numsamples` 的值不大于x_shape[-1]。
        - **ValueError** - 如果 `x` 某一行元素的和小于零。
        - **ValueError** - 如果 `x` 每一行都存在小于零的值。
        - **ValueError** - 如果 `numsamples` 小于等于0。
