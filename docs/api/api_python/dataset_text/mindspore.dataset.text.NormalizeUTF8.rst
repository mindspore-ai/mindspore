mindspore.dataset.text.NormalizeUTF8
====================================

.. py:class:: mindspore.dataset.text.NormalizeUTF8(normalize_form=NormalizeForm.NFKC)

    对UTF-8编码的字符串进行规范化处理。

    .. note::
        Windows平台尚不支持 `NormalizeUTF8` 。

    参数：
        - **normalize_form** (:class:`~.text.NormalizeForm`, 可选) - 想要使用的规范化模式。可选值详见 :class:`~.text.NormalizeForm` 。
          默认值： ``NormalizeForm.NFKC`` 。

    异常：
        - **TypeError** - 当 `normalize_form` 不为 :class:`~.text.NormalizeForm` 类型。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
