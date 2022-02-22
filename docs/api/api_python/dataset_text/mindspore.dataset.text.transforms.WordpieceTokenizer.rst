mindspore.dataset.text.transforms.WordpieceTokenizer
====================================================

.. py:class:: mindspore.dataset.text.transforms.WordpieceTokenizer(vocab, suffix_indicator='##', max_bytes_per_token=100,unknown_token='[UNK]', with_offsets=False)

    对输入的1-D字符串分词为子词(subword)标记。

    **参数：**

    - **vocab** (Vocab) - 词汇表对象。
    - **suffix_indicator** (str, 可选) - 用来表示子词是某个词的一部分，默认值：'##'。
    - **max_bytes_per_token** (int，可选) - 指定最长标记(token)的长度，超过此长度的标记将不会被进一步拆分。
    - **unknown_token** (str，可选) - 当词表中无法找到某个标记(token)时，将替换为此参数的值。
      如果此参数为空字符串，则直接返回该标记。默认值：'[UNK]'。
    - **with_offsets** (bool, 可选) - 是否输出标记(token)的偏移量，默认值：False。

    **异常：**

    - **TypeError** - 参数 `vocab` 的类型不为text.Vocab。
    - **TypeError** - 参数 `suffix_indicator` 的类型不为string。
    - **TypeError** - 参数 `max_bytes_per_token` 的类型不为int。
    - **TypeError** - 参数 `unknown_token` 的类型不为string。
    - **TypeError** - 参数 `with_offsets` 的类型不为bool。
    - **ValueError** - 参数 `max_bytes_per_token` 的值为负数。
