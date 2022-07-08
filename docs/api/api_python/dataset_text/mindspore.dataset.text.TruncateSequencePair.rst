mindspore.dataset.text.TruncateSequencePair
===========================================

.. py:class:: mindspore.dataset.text.TruncateSequencePair(max_length)

    截断一对 1-D 字符串的内容，使其总长度小于给定长度。

    TruncateSequencePair接收两个Tensor作为输入并返回两个Tensor作为输出。

    参数：
        - **max_length** (int) - 最大截断长度。

    异常：
        - **TypeError** - 参数 `max_length` 的类型不是int。
