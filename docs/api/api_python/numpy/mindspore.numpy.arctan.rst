mindspore.numpy.arctan
=================================

.. py:function:: mindspore.numpy.arctan(x, dtype=None)

    逐元素计算反正切函数。
    这是正切函数的反函数，如果 :math:`y=\tan(x)` ，则 :math:`x=\arctan(y)` 。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。如果 ``x`` 是标量，则返回结果为标量。