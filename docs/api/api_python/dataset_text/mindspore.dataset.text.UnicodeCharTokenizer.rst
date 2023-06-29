mindspore.dataset.text.UnicodeCharTokenizer
===========================================

.. py:class:: mindspore.dataset.text.UnicodeCharTokenizer(with_offsets=False)

    使用Unicode分词器将字符串分词为Unicode字符。

    参数：
        - **with_offsets** (bool, 可选) - 是否输出标记(token)的偏移量。默认值： ``False`` 。

    异常：
        - **TypeError** - 参数 `with_offsets` 的类型不为bool。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
