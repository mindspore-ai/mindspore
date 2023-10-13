mindspore.dataset.text.RegexReplace
===================================

.. py:class:: mindspore.dataset.text.RegexReplace(pattern, replace, replace_all=True)

    使用正则表达式将输入 UTF-8 字符串的部分字符串替换为指定文本。

    .. note:: Windows平台尚不支持 `RegexReplace` 。

    参数：
        - **pattern** (str) - 正则表达式，即用来描述或匹配一系列符合某个句法规则的字符串。
        - **replace** (str) - 用来替换匹配元素的字符串。
        - **replace_all** (bool, 可选) - 是否替换全部匹配元素。若为 ``False`` ，则只替换第一个成功匹配的元素；
          否则，替换所有匹配的元素。默认值： ``True`` 。

    异常：
        - **TypeError** - 当 `pattern` 不为str类型。
        - **TypeError** - 当 `replace` 不为str类型。
        - **TypeError** - 当 `replace_all` 不为bool类型。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
