mindspore.ops.Fill
==================

.. py:class:: mindspore.ops.Fill(*args, **kwargs)

    创建一个填充了Scalar值的Tensor。shape由 `shape` 参数指定，并用`value` 值填充该Tensor。

    **输入：**

    - **type** (mindspore.dtype) - 指定输出Tensor的数据类型。只支持常量值。
    - **shape** (tuple) - 指定输出Tensor的shape。只支持常量值。
    - **value** (scalar) - 用来填充输出Tensor的值。只支持常量值。

    **输出：**

    Tensor，shape为 `shape` 的值，值为 `value` 。

    **异常：**

    **TypeError** - `shape` 不是元组。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> fill = ops.Fill()
    >>> output = fill(mindspore.float32, (2, 2), 1)
    >>> print(output)
    [[1. 1.]
     [1. 1.]]
    >>> output = fill(mindspore.float32, (3, 3), 0)
    >>> print(output)
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    