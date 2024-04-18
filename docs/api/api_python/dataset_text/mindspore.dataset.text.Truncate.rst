mindspore.dataset.text.Truncate
===============================

.. py:class:: mindspore.dataset.text.Truncate(max_seq_len)

    截断输入序列，使其不超过最大长度。

    参数：
        - **max_seq_len** (int) - 最大截断长度。

    异常：
        - **TypeError** - 如果 `max_seq_len` 的类型不是int。
        - **ValueError** - 如果 `max_seq_len` 的值小于或等于0。
        - **RuntimeError** - 如果输入张量的数据类型不是bool、int、float、double或者str。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
