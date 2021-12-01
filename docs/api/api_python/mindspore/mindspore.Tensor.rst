mindspore.Tensor
================

.. py:class:: mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None)

    用来存储数据。 继承自C++中的 `Tensor` 对象。有些函数是用C++实现的，有些函数是用Python实现的。

    **参数：**

    - **input_data** (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray])：张量的输入数据。
    - **dtype** (:class:`mindspore.dtype`)：输入数据应是在 `mindspore.dtype` 中定义的None、bool或numeric类型。该参数用于定义输出张量的数据类型。如果值为None，则输出张量的数据类型与 `input_data` 的相同。默认值：None。
    - **shape** (Union[tuple, list, int])：用来表示张量的形状，可以是整数列表、整数元组或单一整数。如果 `input_data` 已经被设置，则不需要再设置 `shape` 。默认值：None。
    - **init** (Initializer)：用来表示初始化数据的信息。init用于在并行模式下的延迟初始化。一般情况下，不建议在其他条件下使用init接口来初始化参数。如果使用init接口来初始化参数，需要调用 `Tensor.init_data` 接口把 `Tensor` 转换为实际数据。

    **返回：**

    Tensor。如果未设置 `dtype` 和 `shape` ，返回与 `input_data` 具有相同数据类型和形状的张量。如果设置了 `dtype` 或 `shape` ，则输出的张量的数据类型或形状与设置的相同。

    **样例：**

    >>> import numpy as np
    >>> import mindspore as ms
    >>> from mindspore import Tensor
    >>> from mindspore.common.initializer import One
    >>> # 用numpy.ndarray初始化张量
    >>> t1 = Tensor(np.zeros([1, 2, 3]), ms.float32)
    >>> assert isinstance(t1, Tensor)
    >>> assert t1.shape == (1, 2, 3)
    >>> assert t1.dtype == ms.float32
    >>>
    >>> # 用float标量初始化张量
    >>> t2 = Tensor(0.1)
    >>> assert isinstance(t2, Tensor)
    >>> assert t2.dtype == ms.float64
    ...
    >>> # 用init初始化张量
    >>> t3 = Tensor(shape = (1, 3), dtype=ms.float32, init=One())
    >>> assert isinstance(t3, Tensor)
    >>> assert t3.shape == (1, 3)
    >>> assert t3.dtype == ms.float32

    .. py:method:: T
        :property:

        返回转置后的张量。

    .. py:method:: abs()

        返回每个元素的绝对值。

        **返回：**

        张量， 含有每个元素的绝对值。

        **支持平台：**

        ``Ascend`` ``GPU`` ``CPU``

        **样例：**

        >>> from mindspore import Tensor
        >>> a = Tensor([1.1, -2.1]).astype("float32")
        >>> output = a.abs()
        >>> print(output)
        [1.1 2.1]

    .. py:method:: all(axis=(), keep_dims=False)

        检查在指定轴上所有元素是否均为True。

        **参数：**

        - **axis** (Union[None, int, tuple(int))：被简化的维度。 当 `axis` 为None或者空元组的时候，简化所有维度。 默认值：()。
        - **keep_dims** (bool)：是否会保留被简化的维度。 默认值： False。

        **返回：**

        Tensor。如果在指定轴方向上所有数组元素都为True，则其值为True，否则其值为False。如果轴为None或空元组，则简化所有维度。

        **支持平台：**

        ``Ascend`` ``GPU`` ``CPU``

        **样例：**

        >>> from mindspore import Tensor
        >>> a = Tensor([True, True, False])
        >>> output = a.all()
        >>> print(output)
        False

    .. py:method:: any(axis=(), keep_dims=False)

        检查在指定轴方向上是否存在任意为True的数组元素。

        **参数：**

        - **axis** (Union[None, int, tuple(int))：简化的维度。当轴为None或空元组时，简化所有维度。默认值：()。
        - **keep_dims** (bool)：表示是否保留简化后的维度。默认值：False。

        **返回：**

        Tensor。如果在指定轴方向上所有数组元素都为True，则其值为True，否则其值为False。如果轴为None或空元组，则简化所有维度。

        **支持平台：**

        ``Ascend`` ``GPU`` ``CPU``

        **样例：**

        >>> from mindspore import Tensor
        >>> a = Tensor([True, True, False])
        >>> output = a.any()
        >>> print(output)
        True

    .. py:method:: asnumpy()

        将张量转换为NumPy数组。

    .. py:method:: dtype
        :property:

        返回张量的数据类型（:class:`mindspore.dtype`）。

    .. py:method:: expand_as(x)

        将目标张量的维度扩展为输入张量的维度。

        **参数：**

        **x** (Tensor)：输入的张量。该张量的形状必须遵守广播规则。

        **返回：**

        Tensor，维度与输入张量的相同。

    .. py:method:: from_numpy(array)
        :static:

        将NumPy数组转换为张量，且不需要复制数据。

        **参数：**

        array (numpy.array)：输入数组。

        **返回：**

        Tensor，与输入的张量具有相同的数据类型。


    .. py:method:: mean(axis=(), keep_dims=False)

        通过计算出维度中的所有元素的平均值来简化张量的维度。

        **参数：**

        - **axis** (Union[None, int, tuple(int), list(int)])：简化的维度。当轴为None或空元组时，简化所有维度。默认值：()。
        - **keep_dims** (bool)：表示是否保留简化后的维度。默认值：False。

        **返回：**

        Tensor，与输入的张量具有相同的数据类型。

        **支持平台：**

        ``Ascend`` ``GPU`` ``CPU``

        **样例：**

        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.array([1, 2, 3], dtype=np.float32))
        >>> output = input_x.mean()
        >>> print(output)
        2.0
