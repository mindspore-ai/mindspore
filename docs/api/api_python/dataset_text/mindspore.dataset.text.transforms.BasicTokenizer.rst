mindspore.dataset.text.transforms.BasicTokenizer
=================================================

.. py:class:: mindspore.dataset.text.transforms.BasicTokenizer(lower_case=False, keep_whitespace=False, normalization_form=NormalizeForm.NONE, preserve_unused_token=True, with_offsets=False)

    按照指定规则对输入的UTF-8字符串进行分词。

    .. note:: Windows平台尚不支持 `BasicTokenizer` 。

    **参数：**

    - **lower_case** (bool，可选) - 若为True，将对输入执行 :class:`mindspore.dataset.text.transforms.CaseFold` 、NFD模式 :class:`mindspore.dataset.text.transforms.NormalizeUTF8` 和 :class:`mindspore.dataset.text.transforms.RegexReplace` 等操作，将文本转换为小写并删除重音字符；若为False，将只执行 `normalization_form` 模式 :class:`mindspore.dataset.text.transforms.NormalizeUTF8` 操作。默认值：False。
    - **keep_whitespace** (bool，可选) - 是否在分词输出中保留空格。默认值：False。
    - **normalization_form** (:class:`mindspore.dataset.text.NormalizeForm`，可选) - 字符串规范化模式，仅当 `lower_case` 为False时生效，取值可为NormalizeForm.NONE、NormalizeForm.NFC、NormalizeForm.NFKC、NormalizeForm.NFD或NormalizeForm.NFKD。默认值：NormalizeForm.NONE。

      - NormalizeForm.NONE：对输入字符串不做任何处理。
      - NormalizeForm.NFC：对输入字符串进行C形式规范化。
      - NormalizeForm.NFKC：对输入字符串进行KC形式规范化。
      - NormalizeForm.NFD：对输入字符串进行D形式规范化。
      - NormalizeForm.NFKD：对输入字符串进行KD形式规范化。

    - **preserve_unused_token** (bool，可选) - 若为True，将不会对特殊词汇进行分词，如 '[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]' 等。默认值：True。
    - **with_offsets** (bool，可选) - 是否输出词汇在字符串中的偏移量。默认值：False。
