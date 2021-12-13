mindspore.nn.Flatten
====================

.. py:class:: mindspore.nn.Flatten

    对输入Tensor的第0维的batch size之外的维度进行展平操作。

    **输入：**

    - **x** (Tensor) - 要展平的输入Tensor。shape为 :math:`(N,*)`，其中 :math:`*` 表示任意的附加维度数。数据类型为Number。

    **输出：**

    Tensor，shape为 :math:`(N,X)`，其中 :math:`X` 是输入 `x` 的shape除N之外的其余维度的乘积。

    **异常：**

    **TypeError** - `x` 不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([[[1.2, 1.2], [2.1, 2.1]], [[2.2, 2.2], [3.2, 3.2]]]), mindspore.float32)
    >>> net = nn.Flatten()
    >>> output = net(x)
    >>> print(output)
    [[1.2 1.2 2.1 2.1]
     [2.2 2.2 3.2 3.2]]
    >>> print(f"Before flatten the x shape is {x.shape}.")
    展平前x的shape为(2, 2, 2)
    >>> print(f"After flatten the output shape is {output.shape}.")
    展平后输出的shape为(2, 4)
    