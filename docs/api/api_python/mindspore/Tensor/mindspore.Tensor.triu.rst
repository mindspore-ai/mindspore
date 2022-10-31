mindspore.Tensor.triu
=====================

.. py:method:: mindspore.Tensor.triu(diagonal=0)

    根据对角线返回相应的三角矩阵。默认为主对角线。

    参数：
        - **diagonal** (int) - 对角线的系数。默认值：0。

    返回：
        Tensor，shape和dtype与输入相同。

    异常：
        - **TypeError** - 如果 `diagonal` 不是int。
        - **TypeError** - 如果 `x` 不是Tensor。
        - **ValueError** - 如果shape的长度小于1。


