mindspore.ops.Eps
=================

.. py:class:: mindspore.ops.Eps(*args, **kwargs)

    创建一个填充 `x` 数据类型最小值的Tensor。

    **输入：**

    - **x** (Tensor) - 用于获取其数据类型最小值的Tensor。数据类型必须为float16或float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape，填充了 `x` 数据类型的最小值。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor([4, 1, 2, 3], mindspore.float32)
    >>> output = ops.Eps()(x)
    >>> print(output)
    [1.5258789e-05 1.5258789e-05 1.5258789e-05 1.5258789e-05]
