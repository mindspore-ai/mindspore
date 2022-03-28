mindspore.dataset.text.transforms.CaseFold
==========================================

.. py:class:: mindspore.dataset.text.transforms.CaseFold()

    将UTF-8编码字符串中的字符规范化为小写，相比 :func:`str.lower` 支持更多字符。

    支持的输入规范化形式详见 `ICU_Normalizer2 <https://unicode-org.github.io/icu-docs/apidoc/released/icu4c/classicu_1_1Normalizer2.html>`_ 。

    .. note:: Windows平台尚不支持 `CaseFold` 。
