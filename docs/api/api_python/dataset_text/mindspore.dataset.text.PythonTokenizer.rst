mindspore.dataset.text.PythonTokenizer
======================================

.. py:class:: mindspore.dataset.text.PythonTokenizer(tokenizer)

    使用用户自定义的分词器对输入字符串进行分词。

    参数：
        - **tokenizer** (Callable) - Python可调用对象，要求接收一个string参数作为输入，并返回一个包含多个string的列表作为返回值。

    异常：
        - **TypeError** - 参数 `tokenizer` 不是一个可调用的Python对象。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
