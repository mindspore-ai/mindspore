mindspore.dataset.text.ToNumber
===============================

.. py:class:: mindspore.dataset.text.ToNumber(data_type)

    将字符串的每个元素转换为数字。

    字符串根据以下链接中指定的规则进行转换，除了任何表示负数的字符串不能转换为无符号整数类型外，规则链接如下：
    https://en.cppreference.com/w/cpp/string/basic_string/stof，
    https://en.cppreference.com/w/cpp/string/basic_string/stoul。

    参数：
        - **data_type** (mindspore.dtype) - 要转换为的数值类型，需要是在 :class:`mindspore.dtype` 定义的数值类型。

    异常：
        - **TypeError** - 参数 `data_type` 不是 :class:`mindspore.dtype` 类型。
        - **RuntimeError** - 字符串类型转换失败，或类型转换时出现溢出。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
