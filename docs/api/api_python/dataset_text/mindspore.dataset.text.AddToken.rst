mindspore.dataset.text.AddToken
===============================

.. py:class:: mindspore.dataset.text.AddToken(token, begin=True)

    将分词(token)添加到序列的开头或结尾处。

    参数：
        - **token** (str) - 待添加的分词(token)。
        - **begin** (bool, 可选) - 选择分词(token)插入的位置，若为True则在序列开头插入，否则在序列结尾插入。默认值：True。

    异常：
        - **TypeError** - 如果 `token` 的类型不为str。
        - **TypeError** - 如果 `begin` 的类型不为bool。
