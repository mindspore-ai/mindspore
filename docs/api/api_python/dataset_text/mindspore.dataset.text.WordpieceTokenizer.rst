mindspore.dataset.text.WordpieceTokenizer
=========================================

.. py:class:: mindspore.dataset.text.WordpieceTokenizer(vocab, suffix_indicator='##', max_bytes_per_token=100, unknown_token='[UNK]', with_offsets=False)

    将输入的字符串切分为子词。

    参数：
        - **vocab** (:class:`mindspore.dataset.text.Vocab`) - 用于查词的词汇表。
        - **suffix_indicator** (str, 可选) - 用于指示子词后缀的前缀标志。默认值：'##'。
        - **max_bytes_per_token** (int，可选) - 分词最大长度，超过此长度的词汇将不会被拆分。默认值：100。
        - **unknown_token** (str，可选) - 对未知词汇的分词输出。当设置为空字符串时，直接返回对应未知词汇作为分词输出；否则，返回该字符串作为分词输出。默认值：'[UNK]'。
        - **with_offsets** (bool, 可选) - 是否输出词汇在字符串中的偏移量。默认值：False。

    异常：
        - **TypeError** - 当 `vocab` 不为 :class:`mindspore.dataset.text.Vocab` 类型。
        - **TypeError** - 当 `suffix_indicator` 的类型不为str。
        - **TypeError** - 当 `max_bytes_per_token` 的类型不为int。
        - **TypeError** - 当 `unknown_token` 的类型不为str。
        - **TypeError** - 当 `with_offsets` 的类型不为bool。
        - **ValueError** - 当 `max_bytes_per_token` 为负数。
