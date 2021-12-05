mindspore.nn.probability.bijector.PowerTransform
=================================================

.. py:class:: mindspore.nn.probability.bijector.PowerTransform(power=0.0, name='PowerTransform')

    乘方Bijector（Power Bijector）。
    此Bijector执行如下操作：

    .. math::
        Y = g(X) = (1 + X * c)^{1 / c}, X >= -1 / c

    其中幂c >= 0。

    Power Bijector将输入从 `[-1/c, inf]` 映射到 `[0, inf]` 。

    当 `c=0` 时，此Bijector等于Exp Bijector。

    **参数：**

    - **power** (float, list, numpy.ndarray, Tensor) - 比例因子。默认值：0。
    - **name** (str) - Bijector名称。默认值：'PowerTransform'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note::
        `power` 的数据类型必须为float。

    **异常：**

    - **ValueError** - `power` 小于0或静态未知。
    - **TypeError** - `power` 的数据类型不是float。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.bijector as msb
    >>> from mindspore import Tensor
    >>> # 初始化PowerTransform Bijector。
    >>> powertransform = msb.PowerTransform(0.5)
    >>> value = Tensor([1, 2, 3], dtype=mindspore.float32)
    >>> ans1 = powertransform.forward(value)
    >>> print(ans1.shape)
    (3,)
    >>> ans2 = powertransform.inverse(value)
    >>> print(ans2.shape)
    (3,)
    >>> ans3 = powertransform.forward_log_jacobian(value)
    >>> print(ans3.shape)
    (3,)
    >>> ans4 = powertransform.inverse_log_jacobian(value)
    >>> print(ans4.shape)
    (3,)

