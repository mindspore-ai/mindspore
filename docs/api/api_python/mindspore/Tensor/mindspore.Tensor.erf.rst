mindspore.Tensor.erf
====================

.. py:method:: mindspore.Tensor.erf()

    逐元素计算原Tensor的高斯误差函数。
    更多细节参考 :func:`mindspore.ops.erf`。

    返回：
        Tensor，具有与原Tensor相同的数据类型和shape。

    异常：
        - **TypeError** - 原Tensor的数据类型既不是float16也不是float32。