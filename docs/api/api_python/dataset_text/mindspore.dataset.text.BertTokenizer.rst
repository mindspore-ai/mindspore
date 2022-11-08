mindspore.dataset.text.BertTokenizer
====================================

.. py:class:: mindspore.dataset.text.BertTokenizer(vocab, suffix_indicator='##', max_bytes_per_token=100, unknown_token='[UNK]', lower_case=False, keep_whitespace=False, normalization_form=NormalizeForm.NONE, preserve_unused_token=True, with_offsets=False)

    使用Bert分词器对字符串进行分词。

    .. note:: Windows平台尚不支持 `BertTokenizer` 。

    参数：
        - **vocab** (:class:`mindspore.dataset.text.Vocab`) - 用于查词的词汇表。
        - **suffix_indicator** (str，可选) - 用于指示子词后缀的前缀标志。默认值：'##'。
        - **max_bytes_per_token** (int，可选) - 分词最大长度，超过此长度的词汇将不会被拆分。默认值：100。
        - **unknown_token** (str，可选) - 对未知词汇的分词输出。当设置为空字符串时，直接返回对应未知词汇作为分词输出；否则，返回该字符串作为分词输出。默认值：'[UNK]'。
        - **lower_case** (bool，可选) - 是否对字符串进行小写转换处理。若为True，会将字符串转换为小写并删除重音字符；若为False，将只对字符串进行规范化处理，其模式由 `normalization_form` 指定。默认值：False。
        - **keep_whitespace** (bool，可选) - 是否在分词输出中保留空格。默认值：False。
        - **normalization_form** (:class:`mindspore.dataset.text.NormalizeForm`，可选) - `Unicode规范化模式 <http://unicode.org/reports/tr15/>`_ ，仅当 `lower_case` 为False时生效，取值可为NormalizeForm.NONE、NormalizeForm.NFC、NormalizeForm.NFKC、NormalizeForm.NFD或NormalizeForm.NFKD。默认值：NormalizeForm.NONE。

          - NormalizeForm.NONE：不进行规范化处理。
          - NormalizeForm.NFC：先以标准等价方式分解，再以标准等价方式重组。
          - NormalizeForm.NFKC：先以兼容等价方式分解，再以标准等价方式重组。
          - NormalizeForm.NFD：以标准等价方式分解。
          - NormalizeForm.NFKD：以兼容等价方式分解。

        - **preserve_unused_token** (bool，可选) - 是否保留特殊词汇。若为True，将不会对特殊词汇进行分词，如 '[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]' 等。默认值：True。
        - **with_offsets** (bool，可选) - 是否输出词汇在字符串中的偏移量。默认值：False。

    异常：
        - **TypeError** - 当 `vocab` 的类型不为 :class:`mindspore.dataset.text.Vocab` 。
        - **TypeError** - 当 `suffix_indicator` 的类型不为str。
        - **TypeError** - 当 `max_bytes_per_token` 的类型不为int。
        - **ValueError** - 当 `max_bytes_per_token` 为负数。
        - **TypeError** - 当 `unknown_token` 的类型不为str。
        - **TypeError** - 当 `lower_case` 的类型不为bool。
        - **TypeError** - 当 `keep_whitespace` 的类型不为bool。
        - **TypeError** - 当 `normalization_form` 的类型不为 :class:`mindspore.dataset.text.NormalizeForm` 。
        - **TypeError** - 当 `preserve_unused_token` 的类型不为bool。
        - **TypeError** - 当 `with_offsets` 的类型不为bool。
