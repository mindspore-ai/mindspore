mindspore.dataset.text.transforms.BasicTokenizer
=================================================

.. py:class:: mindspore.dataset.text.transforms.BasicTokenizer(lower_case=False, keep_whitespace=False, normalization_form=NormalizeForm.NONE, preserve_unused_token=True, with_offsets=False)

       通过特定规则标记UTF-8字符串的标量Tensor。

        **注：**
            Windows平台尚不支持BasicTokenizer。

        **参数：**
        - **lower_case** (bool，可选) - 如果为True，则在输入文本上应用CaseFold、 `NFD` 模式下的NormalizeUTF8、RegexReplace操作，以将文本折叠到较低的用例并删除重音字符。如果为False，则仅在输入文本上应用指定模式下的NormalizeUTF8操作（默认为False）。
        - **keep_whitespace** (bool，可选) - 如果为True，则把空白字符保留在输出标记中，默认值：False。
        - **normalization_form** (NormalizeForm，可选) - 用于指定归一化模式，默认值：NormalizeForm.NONE。这仅在 `lower_case` 为False时有效。可选值为NormalizeForm.NONE、NormalizeForm.NFC、NormalizeForm.NFKC、NormalizeForm.NFD和NormalizeForm.NFKD。

          - NormalizeForm.NONE：对输入字符串不做任何处理。
          - NormalizeForm.NFC：对输入字符串进行C形式规范化。
          - NormalizeForm.NFKC：对输入字符串进行KC形式规范化。
          - NormalizeForm.NFD：对输入字符串进行D形式规范化。
          - NormalizeForm.NFKD：对输入字符串进行KD形式规范化。

        - **preserve_unused_token** (bool，可选) - 如果为True，则不要拆分特殊标记，如'[CLS]'、'[SEP]'、'[UNK]'、'[PAD]'和'[MASK]'，默认值：True。
        - **with_offsets** (bool，可选) - 表示是否输出标记的偏移量，默认值：False。