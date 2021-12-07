mindspore.ops.Gather
======================

.. py:class:: mindspore.ops.Gather(*args, **kwargs)

    返回输入Tensor在指定 `axis` 上 `input_indices` 索引对应的元素组成的切片。

    **输入：**

    - **input_params** (Tensor) - 原始Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
    - **input_indices** (Tensor) - 要切片的索引Tensor，shape为 :math:`(y_1, y_2, ..., y_S)` 。指定原始Tensor中要切片的索引。其值必须在 `[0, input_param.shape[axis])` 范围内，该校验仅在CPU上生效。在Ascend和GPU上超出该范围时，对应的值会置为0。数据类型可以是int32或int64。
    - **axis** (int) - 指定要切片的维度索引。

    **输出：**

    Tensor，shape为 :math:`input\_params.shape[:axis] + input\_indices.shape + input\_params.shape[axis + 1:]` 。

    **异常：**

    - **TypeError** - `axis` 不是int。
    - **TypeError** - `input_params` 或 `input_indices` 不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
    >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
    >>> axis = 1
    >>> output = ops.Gather()(input_params, input_indices, axis)
    >>> print(output)
    [[ 2.  7.]
     [ 4. 54.]
     [ 2. 55.]]
    >>> axis = 0
    >>> output = ops.Gather()(input_params, input_indices, axis)
    >>> print(output)
    [[3. 4. 54. 22.]
     [2. 2. 55.  3.]]
    