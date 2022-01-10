mindspore.ops.OnesLike
======================

.. py:class:: mindspore.ops.OnesLike()

    返回值为1的Tensor，shape和数据类型与输入相同。

    **输入：**

    - **input_x** (Tensor) - 任意维度的Tensor。

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
    