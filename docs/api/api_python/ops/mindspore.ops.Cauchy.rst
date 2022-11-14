mindspore.ops.Cauchy
====================

.. py:class:: mindspore.ops.Cauchy

    从柯西分布中提取的随机数创建shape由 `size` 决定的Tensor。

    .. math::
        \f(x)= \frac{1}{\pi} \frac{\sigma}{(x-median)^2 +\sigma^2}

    输入：
        - **size** (list[int]) - 描述输出Tensor的shape。
        - **sigma** (float) - 位置参数，指定分布峰值的位置。默认值：1.0。
        - **median** (float) - 尺度参数，指定半宽半最大值处的scale参数。默认值：0.0。

    输出：
        Tensor，数据类型为float32，shape由 `size` 决定的Tensor。Tensor中的数值符合柯西分布。

    异常：
        - **TypeError** - `sigma` 不是float。
        - **TypeError** - `median` 不是float。
        - **ValueError** - `size` 不是list。
        - **ValueError** - `size` 是空的。
        - **ValueError** - `size` 中的数值不是正数。
