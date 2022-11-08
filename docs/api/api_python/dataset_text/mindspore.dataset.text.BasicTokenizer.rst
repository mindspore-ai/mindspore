mindspore.dataset.text.BasicTokenizer
======================================

.. py:class:: mindspore.dataset.text.BasicTokenizer(lower_case=False, keep_whitespace=False, normalization_form=NormalizeForm.NONE, preserve_unused_token=True, with_offsets=False)

    按照指定规则对输入的UTF-8编码字符串进行分词。

    .. note:: Windows平台尚不支持 `BasicTokenizer` 。

    参数：
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
        - **TypeError** - 当 `lower_case` 的类型不为bool。
        - **TypeError** - 当 `keep_whitespace` 的类型不为bool。
        - **TypeError** - 当 `normalization_form` 的类型不为 :class:`mindspore.dataset.text.NormalizeForm` 。
        - **TypeError** - 当 `preserve_unused_token` 的类型不为bool。
        - **TypeError** - 当 `with_offsets` 的类型不为bool。
        - **RuntimeError** - 当输入Tensor的数据类型不为str。
