mindspore.ops.ZerosLike
=======================

.. py:class:: mindspore.ops.ZerosLike(*args, **kwargs)

    创建新的Tensor。它的所有元素的值都为0。

    返回具有与输入Tensor相同shape和数据类型的值为0的Tensor。

    **输入：**

    - **input_x** (Tensor) - 输入Tensor。数据类型为int32、int64、float16或float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `input_x` 相同的shape和数据类型，并填充了0。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> zeroslike = ops.ZerosLike()
    >>> input_x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    >>> output = zeroslike(input_x)
    >>> print(output)
    [[0. 0.]
     [0. 0.]]
    