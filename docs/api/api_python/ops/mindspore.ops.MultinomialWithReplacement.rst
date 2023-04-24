mindspore.ops.MultinomialWithReplacement
========================================

.. py:class:: mindspore.ops.MultinomialWithReplacement(numsamples, replacement=False)

    返回一个Tensor，其中每行包含从重复采样的多项式分布中抽取的 `numsamples` 个索引。与 `Multinomial` 不同， `MultinomialWithReplacement` 允许多次选择相同的结果。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.multinomial_with_replacement`。

    .. note::
        输入的行不需要求和为1（在这种情况下，使用值作为权重），但必须是非负的、有限的，并且具有非零和。

    参数：
        - **numsamples** (int) - 抽取样本量，必须大于零。
        - **replacement** (bool，可选) - 是否有放回地抽取。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 包含概率的累积和的输入Tensor，必须为一维或二维。
        - **seed** (Tensor) - 如果将随机种子设置为-1，并将 `offset` 设置为0，则随机数生成器将使用随机种子进行种植。否则，将使用给定的随机数种子。支持的dtype：int64。
        - **offset** (int) - 为避免种子冲突设置的偏移量。支持的dtype：int64。

    输出：
        Tensor，具有与输入相同的行。每行的采样索引数为 `numsamples` 。
