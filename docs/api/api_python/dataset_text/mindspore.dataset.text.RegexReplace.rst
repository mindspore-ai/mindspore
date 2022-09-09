mindspore.dataset.text.RegexReplace
===================================

.. py:class:: mindspore.dataset.text.RegexReplace(pattern, replace, replace_all=True)

    根据正则表达式对UTF-8编码格式的字符串内容进行正则替换。

    有关支持的正则表达式的模式，请参阅 https://unicode-org.github.io/icu/userguide/strings/regexp.html。

    .. note:: Windows平台尚不支持 `RegexReplace` 。

    参数：
        - **pattern** (str) -  正则表达式的模式。
        - **replace** (str) - 替换匹配元素的字符串。
        - **replace_all** (bool, 可选) - 如果为False，只替换第一个匹配的元素；如果为True，则替换所有匹配的元素。默认值：True。

    异常：
        - **TypeError** - 参数 `pattern` 的类型不是str。
        - **TypeError** - 参数 `replace` 的类型不是str。
        - **TypeError** - 参数 `replace_all` 的类型不是bool。
