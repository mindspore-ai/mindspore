mindspore.nn.probability.bijector.ScalarAffine
=================================================

.. py:class:: mindspore.nn.probability.bijector.ScalarAffine(scale=1.0, shift=0.0, name='ScalarAffine')

    标量仿射Bijector（Scalar Affine Bijector）。
    此Bijector执行如下操作：

    .. math::
        Y = a * X + b

    其中a是比例因子，b是移位因子。

    **参数：**

    - **scale** (float, list, numpy.ndarray, Tensor) - 比例因子。默认值：1.0。
    - **shift** (float, list, numpy.ndarray, Tensor) - 移位因子。默认值：0.0。
    - **name** (str) - Bijector名称。默认值：'ScalarAffine'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note::
        `shift` 和 `scale` 的数据类型必须为float。如果 `shift` 、 `scale` 作为numpy.ndarray或Tensor传入，则它们必须具有相同的数据类型，否则将引发错误。

    **异常：**

    - **TypeError** - `shift` 或 `scale` 的数据类型不为float，或 `shift` 和 `scale` 的数据类型不相同。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> from mindspore import Tensor
    >>>
    >>> # 初始化ScalarAffine Bijector，scale设置为1.0，shift设置为2.0。
    >>> scalaraffine = nn.probability.bijector.ScalarAffine(1.0, 2.0)
    >>> value = Tensor([1, 2, 3], dtype=mindspore.float32)
    >>> ans1 = scalaraffine.forward(value)
    >>> print(ans1.shape)
    (3,)
    >>> ans2 = scalaraffine.inverse(value)
    >>> print(ans2.shape)
    (3,)
    >>> ans3 = scalaraffine.forward_log_jacobian(value)
    >>> print(ans3.shape)
    ()
    >>> ans4 = scalaraffine.inverse_log_jacobian(value)
    >>> print(ans4.shape)
    ()

