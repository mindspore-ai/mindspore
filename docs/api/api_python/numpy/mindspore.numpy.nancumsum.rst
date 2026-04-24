mindspore.numpy.nancumsum
=========================

.. py:function:: mindspore.numpy.nancumsum(a, axis=None, dtype=None)

    返回给定轴上数组元素的累积和，将NaN(非数值)视为零。

    遇到NaN时，累积和不变，数组中的NaN将替换为零。
    对于全是NaN或为空的切片，返回零。

    .. note::
        如果 ``a.dtype`` 是 ``int8`` , ``int16`` 或 ``bool`` ，结果的 `dtype` 将提升为 ``int32`` 。

    参数：
        - **a** (Tensor) - 输入Tensor。
        - **axis** (int, 可选) - 计算累积和所沿轴。若取默认值(None)，将在展平的数组上计算累积和。
        - **dtype** (mindspore.dtype, 可选) - 如果未指定，则与 `a` 相同，除非 `a` 具有精度低于默认平台整数的整数 `dtype` 。在这种情况下，使用默认平台整数。默认值: ``None`` 。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入参数的类型未在上述范围内。
        - **ValueError** - 如果 `axis` 超出范围。