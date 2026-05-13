mindspore.numpy.copysign
========================

.. py:function:: mindspore.numpy.copysign(x1, x2, dtype=None)

    将 `x1` 的符号更改为 `x2` 的符号，逐元素执行。
    如果 `x2` 是一个标量，则它的符号会被复制到 `x1` 的所有元素。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。 不支持复数输入。

    参数：
        - **x1** (Union[int, float, list, tuple, Tensor]) - 需要更改符号的值。
        - **x2** (Union[int, float, list, tuple, Tensor]) - x1的符号将更改为x2的符号。如果 ``x1.shape != x2.shape`` ，它们必须能广播到一个共同的shape(即输出的shape)。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。更改为与 `x2` 相同符号的 `x1` 的值。如果 `x1` 与 `x2` 都是标量，则结果也是标量。

    异常：
        - **TypeError** - 如果输入的dtype不在规定的类型内或输入不能被转换为Tensor。