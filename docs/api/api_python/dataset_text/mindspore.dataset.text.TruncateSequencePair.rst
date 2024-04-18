mindspore.dataset.text.TruncateSequencePair
===========================================

.. py:class:: mindspore.dataset.text.TruncateSequencePair(max_length)

    对两列 1-D 字符串输入进行截断，使其总长度小于指定长度。

    参数：
        - **max_length** (int) - 字符串最大输出总长。当其大于或等于两列输入字符串总长时，不进行截断；
          否则，优先截取两列输入中的较长者，直至其总长等于该值。

    异常：
        - **TypeError** - 当 `max_length` 不为int类型。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
