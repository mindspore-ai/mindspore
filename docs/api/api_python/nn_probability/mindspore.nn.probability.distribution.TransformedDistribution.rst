mindspore.nn.probability.distribution.TransformedDistribution
==============================================================

.. py:class:: mindspore.nn.probability.distribution.TransformedDistribution(bijector, distribution, seed=None, name='transformed_distribution')

    转换分布（Transformed Distribution）。
    该类包含一个Bijector和一个分布，并通过Bijector定义的操作将原始分布转换为新分布。可如果原始分布为 :math:`X` ，Bijector的映射函数为 :math:`g`， 那么对应的转换分布为 :math:`Y = g(X)` 。


    **参数：**

    - **bijector** (Bijector) - 要执行的转换。
    - **distribution** (Distribution) - 原始分布。必须具有float数据类型。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。如果在初始化TransformedDistribution对象时给出了此种子，则对象的采样函数将使用此种子；否则，将使用基础分布的种子。
    - **name** (str) - 转换分布的名称。默认值：'transformed_distribution'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        用于初始化原始分布的参数不能为None。例如，由于未指定 `mean` 和 `sd` ，因此无法使用mynormal = msd.Normal(dtype=mindspore.float32)初始化TransformedDistribution。

    **异常：**

    - **ValueError** - bijector不是Bijector类，distribution不是Distribution类。

    **样例：**

    >>> import numpy as np
    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> import mindspore.nn.probability.bijector as msb
    >>> from mindspore import Tensor
    >>> class Net(nn.Cell):
    ...     def __init__(self, shape, dtype=mindspore.float32, seed=0, name='transformed_distribution'):
    ...         super(Net, self).__init__()
    ...         # 创建转换分布
    ...         self.exp = msb.Exp()
    ...         self.normal = msd.Normal(0.0, 1.0, dtype=dtype)
    ...         self.lognormal = msd.TransformedDistribution(self.exp, self.normal, seed=seed, name=name)
    ...         self.shape = shape
    ...
    ...     def construct(self, value):
    ...         cdf = self.lognormal.cdf(value)
    ...         sample = self.lognormal.sample(self.shape)
    ...         return cdf, sample
    >>> shape = (2, 3)
    >>> net = Net(shape=shape, name="LogNormal")
    >>> x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    >>> tx = Tensor(x, dtype=mindspore.float32)
    >>> cdf, sample = net(tx)
    >>> print(sample.shape)
    (2, 3)

