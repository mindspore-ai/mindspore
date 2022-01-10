mindspore.nn.probability.distribution.LogNormal
================================================

.. py:class:: mindspore.nn.probability.distribution.LogNormal(loc=None, scale=None, seed=0, dtype=mstype.float32, name='LogNormal')

    对数正态分布（LogNormal distribution）。
    连续随机分布，取值范围为 :math:`(0, \inf)` ，概率密度函数为

    .. math:: 
        f(x, a, b) = 1 / xb\sqrt{2\pi} \exp(-(\ln(x) - a)^2 / 2b^2).

    其中 :math:`a, b` 为分别为基础正态分布的平均值和标准差。
    服从对数正态分布的随机变量的对数服从正态分布。它被构造为正态分布的指数变换。

    **参数：**

    - **loc** (int, float, list, numpy.ndarray, Tensor) - 基础正态分布的平均值。默认值：None。
    - **scale** (int, float, list, numpy.ndarray, Tensor) - 基础正态分布的标准差。默认值：None。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：0。
    - **dtype** (mindspore.dtype) - 分布类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'LogNormal'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        - `scale` 必须大于零。
        - `dtype` 必须是float，因为对数正态分布是连续的。

    **异常：**

    - **ValueError** - `scale` 中元素小于0。
    - **TypeError** - `dtype` 不是float的子类。

    **样例：**

    >>> import numpy as np
    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> class Prob(nn.Cell):
    ...     def __init__(self):
    ...         super(Prob, self).__init__()
    ...         self.ln = msd.LogNormal(np.array([0.3]), np.array([[0.2], [0.4]]), dtype=mindspore.float32)
    ...     def construct(self, x_):
    ...         return self.ln.prob(x_)
    >>> pdf = Prob()
    >>> output = pdf(Tensor([1.0, 2.0], dtype=mindspore.float32))
    >>> print(output.shape)
    (2, 2)

    .. py:method:: loc
        :property:

        返回分布位置。

        **返回：**

        Tensor, 分布的位置值。

    .. py:method:: scale
        :property:

        返回分布比例。

        **返回：**

        Tensor, 分布的比例值。
