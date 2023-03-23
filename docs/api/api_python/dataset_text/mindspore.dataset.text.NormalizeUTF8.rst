mindspore.dataset.text.NormalizeUTF8
====================================

.. py:class:: mindspore.dataset.text.NormalizeUTF8(normalize_form=NormalizeForm.NFKC)

    对UTF-8编码的字符串进行规范化处理。

    .. note:: Windows平台尚不支持 `NormalizeUTF8` 。

    参数：
        - **normalize_form** (:class:`mindspore.dataset.text.NormalizeForm` , 可选) - 指定不同的规范化形式，可以取值为
          NormalizeForm.NONE, NormalizeForm.NFC, NormalizeForm.NFKC、NormalizeForm.NFD、NormalizeForm.NFKD此四种unicode中的
          任何一种形式。默认值：NormalizeForm.NFKC。

          - NormalizeForm.NONE，对输入字符串不做任何处理。
          - NormalizeForm.NFC，对输入字符串进行C形式规范化。
          - NormalizeForm.NFKC，对输入字符串进行KC形式规范化。
          - NormalizeForm.NFD，对输入字符串进行D形式规范化。
          - NormalizeForm.NFKD，对输入字符串进行KD形式规范化。

          有关规范化详细信息，请参阅 http://unicode.org/reports/tr15/。

    异常：
        - **TypeError** - 参数 `normalize_form` 的类型不是 :class:`mindspore.dataset.text.NormalizeForm` 。
