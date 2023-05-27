mindspore.ops.celu
========================

.. py:function:: mindspore.ops.celu(x, alpha=1.0)

    celu激活函数，逐元素计算输入Tensor的celu（Continuously differentiable exponential linear units）值。公式定义如下：

    .. math::
        \text{CeLU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
    
    详情请参考： `celu <https://arxiv.org/abs/1704.07483>`_ 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x** (Tensor) - celu的输入，数据类型为float16或float32。
        - **alpha** (float，可选) - celu公式定义的阈值 :math:`\alpha` 。默认值： ``1.0`` 。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `alpha` 不是float。
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的dtype既不是float16也不是float32。
        - **ValueError** - `alpha` 的值为0。