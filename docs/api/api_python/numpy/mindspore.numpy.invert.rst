mindspore.numpy.invert
======================

.. py:function:: mindspore.numpy.invert(x, dtype=None)

    逐元素计算按位取反或按位非。
    对输入数组中整数的底层二进制表示执行按位非操作。
    该通用函数实现了 C/Python 运算符 ~ 。
    对于有符号的整数输入，返回其二进制补码表示。在二进制补码系统中，负数由其绝对值的二进制补码表示。这是计算机上表示有符号整数的最常见方法。
    一个N位的二进制补码系统可以表示范围从 ``-2^{N-1}`` 到 ``+2^{N-1}-1`` 的所有整数。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。
        在Ascend上支持的数据类型为np.int16和np.uint16。

    参数：
        - **x** (Tensor) - 仅处理int和bool类型。
        - **dtype** (:class:`mindspore.dtype`，可选) - 默认值: ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。