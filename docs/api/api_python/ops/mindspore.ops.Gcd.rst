mindspore.ops.Gcd
=================

.. py:class:: mindspore.ops.Gcd

    逐元素计算输入Tensor的最大公约数。两个输入的shape需要能进行广播操作，并且数据类型必须为：int32、int64。

    输入：
        - **x1** (Tensor) - 第一个输入Tensor。
        - **x2** (Tensor) - 第二个输入Tensor。
    
    输出：
        - Tensor, shape与输入进行广播后的其中一个相同，数据类型为两个输入中的最高精度的类型。

    异常：
        - **TypeError** - 如果 `x1` 或者 `x2` 的数据类型不是int32或者int64。
        - **ValueError** - 如果两个输入的shape不能进行广播操作。
