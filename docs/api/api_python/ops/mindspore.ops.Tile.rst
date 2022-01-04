mindspore.ops.Tile
===================

.. py:class:: mindspore.ops.Tile(*args, **kwargs)

    按照给定的次数复制Tensor。

    通过复制 `multiples` 次 `input_x` 来创建新的Tensor。输出Tensor的第i维度有 `input_x.shape[i] * multiples[i]` 个元素，并且 `input_x` 的值沿第i维度被复制 `multiples[i]` 次。

    .. note::

        `multiples` 的长度必须大于或等于 `input_x` 的维度。

    **输入：**

    - **input_x** (Tensor) - 1-D或更高的Tensor。将输入Tensor的shape设置为 :math:`(x_1, x_2, ..., x_S)` 。
    - **multiples** (tuple[int]) - 输入tuple由多个整数构成，如 :math:`(y_1, y_2, ..., y_S)` 。`multiples` 的长度不能小于 `input_x` 的维度。只支持常量值。

    **输出：**

    Tensor，具有与 `input_x` 相同的数据类型。假设 `multiples` 的长度为 `d` ，`input_x` 的维度为 `input_x.dim`。

    - 如果 `input_x.dim = d`， 将其相应位置的shape相乘，输出的shape为 :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)` 。
    - 如果 `input_x.dim < d`， 在 `input_x` 的shape的前面填充1，直到它们的长度一致。例如将 `input_x` 的shape设置为 :math:`(1, ..., x_1, ..., x_R, x_S)` ，然后可以将其相应位置的shape相乘，输出的shape为 :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)` 。

    **异常：**

    - **TypeError** - `multiples` 不是tuple或者其元素并非全部是int。
    - **ValueError** - `multiples` 的元素并非全部大于0。
    - **ValueError** - `multiples` 的长度小于 `input_x` 中的维度。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> tile = ops.Tile()
    >>> input_x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
    >>> multiples = (2, 3)
    >>> output = tile(input_x, multiples)
    >>> print(output)
    [[1.  2.  1.  2.  1.  2.]
        [3.  4.  3.  4.  3.  4.]
        [1.  2.  1.  2.  1.  2.]
        [3.  4.  3.  4.  3.  4.]]
    >>> multiples = (2, 3, 2)
    >>> output = tile(input_x, multiples)
    >>> print(output)
    [[[1. 2. 1. 2.]
      [3. 4. 3. 4.]
      [1. 2. 1. 2.]
      [3. 4. 3. 4.]
      [1. 2. 1. 2.]
      [3. 4. 3. 4.]]
      [[1. 2. 1. 2.]
      [3. 4. 3. 4.]
      [1. 2. 1. 2.]
      [3. 4. 3. 4.]
      [1. 2. 1. 2.]
      [3. 4. 3. 4.]]]
    
