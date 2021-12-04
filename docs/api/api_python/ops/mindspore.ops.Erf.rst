mindspore.ops.Erf
=================

.. py:class:: mindspore.ops.Erf(*args, **kwargs)

    按元素计算 `x` 的高斯误差函数。

    .. math::

        erf(x)=\frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    **输入：**

    - **x** (Tensor) - 用于计算高斯误差函数的Tensor。数据类型必须为float16或float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度数，其秩应小于8。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape。

    **异常：**

    - **TypeError** - `x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
    >>> erf = ops.Erf()
    >>> output = erf(x)
    >>> print(output)
    [-0.8427168   0.          0.8427168   0.99530876  0.99997765]
    