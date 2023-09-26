mindspore.dataset.text.UnicodeScriptTokenizer
=============================================

.. py:class:: mindspore.dataset.text.UnicodeScriptTokenizer(keep_whitespace=False, with_offsets=False)

    使用UnicodeScript分词器对UTF-8编码的字符串进行分词。

    .. note:: Windows平台尚不支持 `UnicodeScriptTokenizer` 。

    参数：
        - **keep_whitespace** (bool, 可选) - 是否输出空白标记(token)。默认值： ``False`` 。
        - **with_offsets** (bool, 可选) - 是否输出各Token在原字符串中的起始和结束偏移量。默认值： ``False`` 。

    异常：
        - **TypeError** - 参数 `keep_whitespace` 的类型不为bool。
        - **TypeError** - 参数 `with_offsets` 的类型不为bool。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
