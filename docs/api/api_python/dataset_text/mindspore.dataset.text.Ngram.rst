mindspore.dataset.text.Ngram
============================

.. py:class:: mindspore.dataset.text.Ngram(n, left_pad=("", 0), right_pad=("", 0), separator=" ")

    从1-D的字符串生成N-gram。

    关于N-gram是什么以及它是如何工作的，请参阅 `N-gram <https://en.wikipedia.org/wiki/N-gram#Examples>`_ 。

    参数：
        - **n** (list[int]) - n-gram 中的 n，它是一个正整数列表。例如 n=[4, 3]，结果将是Tensor包含一个4-gram和一个3-gram的字符串。
          如果输入的字符不足以构造一个n-gram，则返回一个空字符串。例如在["mindspore", "best"] 应用 3-gram 将导致生成一个空字符串。
        - **left_pad** (tuple, 可选) - 指定序列的左侧填充，传入tuple的形式为 ("pad_token",pad_width)。
          pad_width 的上限值为 `n` -1。例如，指定 `left_pad=("_", 2)` 将用 "__" 填充序列的左侧。默认值：("", 0)。
        - **right_pad** (tuple, 可选) - 指定序列的右侧填充，传入tuple的形式为 ("pad_token", pad_width)。
          pad_width 的上限值为 `n` -1。例如，指定 `right_pad=("_", 2)` 将用 "__" 填充序列的右侧。默认值：("", 0)。
        - **separator** (str, 可选) - 指定用于将字符串连接在一起的分隔符。
          例如，如果对 ["mindspore", "amazing"] 应用 2-gram 并指定分隔符为"-"，结果将是 ["mindspore-amazing"]。默认值：" "，使用空格作为分隔符。

    异常：      
        - **TypeError** - 参数 `n` 包含的值类型不为int。
        - **ValueError** - 参数 `n` 包含的值不为正数。
        - **ValueError** - 参数 `left_pad` 不是一个长度2的Tuple[str, int]。
        - **ValueError** - 参数 `right_pad` 不是一个长度2的Tuple[str, int]。
        - **TypeError** - 参数 `separator` 的类型不是str。
