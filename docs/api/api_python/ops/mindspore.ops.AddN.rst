mindspore.ops.AddN
===================

.. py:class:: mindspore.ops.AddN

    逐元素将所有输入的Tensor相加。

    更多参考详见 :func:`mindspore.ops.addn`。

    输入：
        - **x** (Union(tuple[Tensor], list[Tensor])) - Tensor组成的tuble或list，类型为 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_ 或 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_ 。

    输出：
        Tensor，与 `x` 的每个Tensor具有相同的shape和数据类型。
