mindspore.dataset.text.NormalizeForm
=====================================

.. py:class:: mindspore.dataset.text.NormalizeForm

    :class:`mindspore.dataset.text.transforms.NormalizeUTF8` 的枚举值。

    `Unicode规范化模式 <http://unicode.org/reports/tr15/#Norm_Forms>_` 可选的枚举值包括： `NormalizeForm.NONE` 、 `NormalizeForm.NFC` 、 `NormalizeForm.NFKC` 、 `NormalizeForm.NFD` 和 `NormalizeForm.NFKD` 。

    - **NormalizeForm.NONE** - 对输入字符串不做任何处理。
    - **NormalizeForm.NFC** - 对输入字符串进行C形式规范化。
    - **NormalizeForm.NFKC** - 对输入字符串进行KC形式规范化。
    - **NormalizeForm.NFD** - 对输入字符串进行D形式规范化。
    - **NormalizeForm.NFKD** - 对输入字符串进行KD形式规范化。
    