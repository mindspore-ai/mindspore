mindspore.ops.msort
====================

.. py:function:: mindspore.ops.msort(input)

    将输入Tensor的元素沿其第一个维度按值升序排序。

    ops.msort(t)相当于ops.Sort(axis=0)(t)[0]。另外可以参考 :class:`mindspore.ops.Sort()`。

    参数：
        - **input** (Tensor) - 需要排序的输入，类型必须是float16或者float32。

    返回：
        排序后的Tensor，与输入的shape和dtype一致。

    异常：
        - **TypeError** -  `input` 的类型既不是float16也不是float32。
