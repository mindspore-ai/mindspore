mindspore.Tensor.sub
====================

.. py:method:: mindspore.Tensor.sub(y)

    更多细节参考 :func:`mindspore.ops.sub`。

    参数：
        - **y** (Union[Tensor, number.Number, bool]) - 是一个number.Number、bool值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为输入中精度较高的类型。

    异常：
        - **TypeError** - `y` 不是Tensor、number.Number或bool。

