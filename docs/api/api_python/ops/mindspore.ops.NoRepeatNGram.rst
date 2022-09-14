mindspore.ops.NoRepeatNGram
============================

.. py:class:: mindspore.ops.NoRepeatNGram(ngram_size=1)

    n-grams出现重复，则更新对应n-gram词序列出现的概率。
    
    在beam search过程中，如果连续的 `ngram_size` 个词存在已生成的词序列中，那么之后预测时，将避免再次出现这连续的 `ngram_size` 个词。例如：当 `ngram_size` 为3时，已生成的词序列为[1,2,3,2,3]，则下一个预测的词不会为2，并且 `log_probs` 的值将替换成负FLOAT_MAX。因为连续的3个词2,3,2不会在词序列中出现两次。

    参数：
        - **ngram_size** (int) - 指定n-gram的长度，必须大于0。默认值：1。

    输入：
        - **state_seq** (Tensor) - n-gram词序列。是一个三维Tensor，其shape为： :math:`(batch\_size, beam\_width, m)` 。
        - **log_probs** (Tensor) - n-gram词序列对应出现的概率，是一个三维Tensor，其shape为： :math:`(batch\_size, beam\_width, vocab\_size)` 。当n-gram重复时，log_probs的值将被负FLOAT_MAX替换。

    输出：
        - **log_probs** (Tensor) - 数据类型和shape与输入 `log_probs` 相同。

    异常：
        - **TypeError** - 如果 `ngram_size` 不是int。
        - **TypeError** - 如果 `state_seq` 和 `log_probs` 都不是Tensor。
