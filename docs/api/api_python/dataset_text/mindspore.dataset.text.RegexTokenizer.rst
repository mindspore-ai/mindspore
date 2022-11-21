mindspore.dataset.text.RegexTokenizer
=====================================

.. py:class:: mindspore.dataset.text.RegexTokenizer(delim_pattern, keep_delim_pattern='', with_offsets=False)

    根据正则表达式对字符串进行分词。
    
    有关支持的正则表达式的模式，请参阅 https://unicode-org.github.io/icu/userguide/strings/regexp.html。

    .. note:: Windows平台尚不支持 `RegexTokenizer` 。

    参数：
        - **delim_pattern** (str) - 以正则表达式表示的分隔符，字符串将被正则匹配的分隔符分割。
        - **keep_delim_pattern** (str, 可选) - 如果被 `delim_pattern` 匹配的字符串也能被 `keep_delim_pattern` 匹配，就可以此分隔符作为标记(token)保存。 
          默认值： ''（空字符），即分隔符不会作为输出标记保留。
        - **with_offsets** (bool, 可选) - 是否输出分词标记(token)的偏移量。默认值：False，不输出。

    异常：
        - **TypeError** - 参数 `delim_pattern` 的类型不是str。
        - **TypeError** - 参数 `keep_delim_pattern` 的类型不是str。
        - **TypeError** - 参数 `with_offsets` 的类型不是bool。
