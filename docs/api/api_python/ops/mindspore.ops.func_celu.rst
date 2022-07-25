mindspore.ops.celu
========================

.. py:function:: mindspore.ops.celu(x, alpha=1.0)

    celu激活函数，按输入元素计算输出，公式定义如下：

    .. math::
        \text{CeLU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    参数：
        - **x** (Tensor) - celu的输入，数据类型为float16或float32。
        - **alpha** (float) - celu公式定义的阈值 :math:`\alpha` 。默认值：1.0。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `alpha` 不是float。
        - **ValueError** - `alpha` 的值为零。
        - **TypeError** - `x` 不是tensor。
        - **TypeError** - `x` 的dtype既不是float16也不是float32。