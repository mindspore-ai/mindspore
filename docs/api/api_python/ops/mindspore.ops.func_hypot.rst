mindspore.ops.hypot
====================

.. py:function:: mindspore.ops.hypot(x, lambd=0.5)

    按元素计算以输入张量为直角边的三角形的斜边。
    两个输入的shape需要支持广播，数据类型应为： float32，float64

    参数：
        - **x** (Tensor) - 第一个输入Tesnor。
        - **other** (Tensor) - 第二个输入Tesnor。

    返回：
        Tensor，shape和广播后的形状相同，数据类型为两个输入数据中更高的。

    异常：
        - **TypeError** - 如果 `x` 或 `other` 的类型不是float32或float64。
        - **ValueError** - 两个输入参数的shape不支持广播。
