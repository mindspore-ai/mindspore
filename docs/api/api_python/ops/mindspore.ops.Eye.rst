mindspore.ops.Eye
==================

.. py:class:: mindspore.ops.Eye()

    创建一个主对角线上元素为1，其余元素为0的Tensor。

    **输入：**

    - **n** (int) - 指定返回Tensor的行数。仅支持常量值。
    - **m** (int) - 指定返回Tensor的列数。仅支持常量值。
    - **t** (mindspore.dtype) - 指定返回Tensor的数据类型。数据类型必须是`bool_ <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_或`number <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_。

    **输出：**

    Tensor，主对角线上为1，其余的元素为0。它的shape由 `n` 和 `m` 指定。数据类型由 `t` 指定。

    **异常：**

    - **TypeError** - `m` 或 `n` 不是int。
    - **ValueError** - `m` 或 `n` 小于1。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> eye = ops.Eye()
    >>> output = eye(2, 2, mindspore.int32)
    >>> print(output)
    [[1 0]
     [0 1]]
    >>> print(output.dtype)
    Int32
    >>> output = eye(1, 2, mindspore.float64)
    >>> print(output)
    [[1. 0.]]
    >>> print(output.dtype)
    Float64
    >>> # if wants a anti-diagonal
    >>> anti_diagonal_input = eye(2, 2, mindspore.int32)
    >>> # Note that ReverseV2 only supports "Ascend" at this time
    >>> reverse = ops.ReverseV2([1])
    >>> anti_diagonal_output = reverse(anti_diagonal_input)
    >>> print(anti_diagonal_output)
    [[0 1]
     [1 0]]
    