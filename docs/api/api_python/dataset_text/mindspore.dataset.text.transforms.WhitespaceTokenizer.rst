mindspore.dataset.text.transforms.WhitespaceTokenizer
=====================================================

.. py:class:: mindspore.dataset.text.transforms.WhitespaceTokenizer(with_offsets=False)

    基于ICU4C定义的空白字符（' ', '\\\\t', '\\\\r', '\\\\n'）对输入字符串进行分词。

    .. note:: Windows 平台尚不支持 WhitespaceTokenizer。

    **参数：**

    - **with_offsets** (bool, 可选) - 是否输出标记(token)的偏移量，默认值：False。

    **异常：**

    - **TypeError** - 参数 `with_offsets` 的类型不为bool。
