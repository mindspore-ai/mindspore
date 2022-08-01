mindspore.Tensor.erfc
=====================

.. py:method:: mindspore.Tensor.erfc()

    逐元素计算原Tensor的互补误差函数。
    更多细节参考 :func:`mindspore.ops.erfc`。

    返回：
        Tensor，具有与原Tensor相同的数据类型和shape。

    异常：
        - **TypeError** - 原Tensor的数据类型既不是float16也不是float32。