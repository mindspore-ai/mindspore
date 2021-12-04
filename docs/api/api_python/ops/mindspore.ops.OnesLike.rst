mindspore.ops.OnesLike
======================

.. py:class:: mindspore.ops.OnesLike(*args, **kwargs)

    创建新Tensor。所有元素的值都为1。

    返回填充了Scalar值为1的具有与输入相同shape和数据类型的Tensor。

    **输入：**

    - **input_x** (Tensor) - 输入Tensor。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `input_x` 相同的shape和类型，并填充了1。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> oneslike = ops.OnesLike()
    >>> input_x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
    >>> output = oneslike(input_x)
    >>> print(output)
    [[1 1]
     [1 1]]
    