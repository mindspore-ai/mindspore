mindspore.dataset.text.BasicTokenizer
======================================

.. py:class:: mindspore.dataset.text.BasicTokenizer(lower_case=False, keep_whitespace=False, normalization_form=NormalizeForm.NONE, preserve_unused_token=True, with_offsets=False)

    按照指定规则对输入的UTF-8编码字符串进行分词。

    .. note:: Windows平台尚不支持 `BasicTokenizer` 。

    参数：
        - **lower_case** (bool，可选) - 是否对字符串进行小写转换处理。若为 ``True`` ，会将字符串转换为小写并删除重音字符；若为 ``False`` ，将只对字符串进行规范化处理，其模式由 `normalization_form` 指定。默认值： ``False`` 。
        - **keep_whitespace** (bool，可选) - 是否在分词输出中保留空格。默认值： ``False`` 。
        - **normalization_form** (:class:`~.text.NormalizeForm`, 可选) - 想要使用的规范化模式。可选值详见 :class:`~.text.NormalizeForm` 。
          默认值： ``NormalizeForm.NFKC`` 。
        - **preserve_unused_token** (bool，可选) - 是否保留特殊词汇。若为 ``True`` ，将不会对特殊词汇进行分词，如 '[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]' 等。默认值： ``True`` 。
        - **with_offsets** (bool，可选) - 是否输出各Token在原字符串中的起始和结束偏移量。默认值： ``False`` 。

    异常：
        - **TypeError** - 当 `lower_case` 的类型不为bool。
        - **TypeError** - 当 `keep_whitespace` 的类型不为bool。
        - **TypeError** - 当 `normalization_form` 的类型不为 :class:`~.text.NormalizeForm` 。
        - **TypeError** - 当 `preserve_unused_token` 的类型不为bool。
        - **TypeError** - 当 `with_offsets` 的类型不为bool。
        - **RuntimeError** - 当输入Tensor的数据类型不为str。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
