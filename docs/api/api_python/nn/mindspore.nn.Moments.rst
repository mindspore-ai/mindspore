mindspore.nn.Moments
====================

.. py:class:: mindspore.nn.Moments(axis=None, keep_dims=None)

    计算 `x` 的均值和方差。

    均值和方差是通过聚合 `x` 在 `axis` 上的值来计算的。特别的，如果 `x` 是1-D的Tensor， `axis` 等于0，这相当于计算向量的均值和方差。

    **参数：**

    - **axis** (Union[int, tuple(int), None]) - 沿指定 `axis` 计算均值和方差，值为None时代表计算 `x` 所有值的均值和方差。默认值：None。
    - **keep_dims** (Union[bool, None]) - 如果为True，计算结果会保留 `axis` 的维度，即均值和方差的维度与输入的相同。如果为False或None，则会消减 `axis` 的维度。默认值：None。

    **输入：**

    - **x** (Tensor) - 用于计算均值和方差的Tensor。数据类型仅支持float16和float32。shape为 :math:`(N,*)`， 其中 :math:`*` 表示任意的附加维度数。

    **输出：**

    - **mean** (Tensor) - `x` 在 `axis` 上的均值，数据类型与输入 `x` 相同。
    - **variance** (Tensor) - `x` 在 `axis` 上的方差，数据类型与输入 `x` 相同。

    **异常：**

    - **TypeError** - `axis` 不是int，tuple或None。
    - **TypeError** - `keep_dims` 既不是bool也不是None。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([[[[1, 2, 3, 4], [3, 4, 5, 6]]]]), mindspore.float32)
    >>> net = nn.Moments(axis=0, keep_dims=True)
    >>> output = net(x)
    >>> print(output)
    (Tensor(shape=[1, 1, 2, 4], dtype=Float32, value=
    [[[[ 1.00000000e+00, 2.00000000e+00, 3.00000000e+00, 4.00000000e+00],
       [ 3.00000000e+00, 4.00000000e+00, 5.00000000e+00, 6.00000000e+00]]]]),
    Tensor(shape=[1, 1, 2, 4], dtype=Float32, value=
    [[[[ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]]))
    >>> net = nn.Moments(axis=1, keep_dims=True)
    >>> output = net(x)
    >>> print(output)
    (Tensor(shape=[1, 1, 2, 4], dtype=Float32, value=
    [[[[ 1.00000000e+00, 2.00000000e+00, 3.00000000e+00, 4.00000000e+00],
       [ 3.00000000e+00, 4.00000000e+00, 5.00000000e+00, 6.00000000e+00]]]]),
    Tensor(shape=[1, 1, 2, 4], dtype=Float32, value=
    [[[[ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]]))
    >>> net = nn.Moments(axis=2, keep_dims=True)
    >>> output = net(x)
    >>> print(output)
    (Tensor(shape=[1, 1, 1, 4], dtype=Float32, value=
    [[[[ 2.00000000e+00, 3.00000000e+00, 4.00000000e+00, 5.00000000e+00]]]]),
    Tensor(shape=[1, 1, 1, 4], dtype=Float32, value=
    [[[[ 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00]]]]))
    >>> net = nn.Moments(axis=3, keep_dims=True)
    >>> output = net(x)
    >>> print(output)
    (Tensor(shape=[1, 1, 2, 1], dtype=Float32, value=
    [[[[ 2.50000000e+00],
       [ 4.50000000e+00]]]]), Tensor(shape=[1, 1, 2, 1], dtype=Float32, value=
    [[[[ 1.25000000e+00],
       [ 1.25000000e+00]]]]))
    