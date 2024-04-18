mindspore.dataset.text.UnicodeCharTokenizer
===========================================

.. py:class:: mindspore.dataset.text.UnicodeCharTokenizer(with_offsets=False)

    对输入字符串中的Unicode字符进行分词。

    参数：
        - **with_offsets** (bool, 可选) - 是否输出各Token在原字符串中的起始和结束偏移量。默认值： ``False`` 。

    异常：
        - **TypeError** - 当 `with_offsets` 不为bool类型。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
