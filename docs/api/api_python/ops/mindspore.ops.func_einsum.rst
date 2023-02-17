mindspore.ops.einsum
====================

.. py:function:: mindspore.ops.einsum(equation, *operands)

    基于爱因斯坦求和约定（Einsum）符号，沿着指定维度对输入Tensor元素的乘积求和。你可以使用这个运算符来执行对角线、减法、转置、矩阵乘法、乘法、内积运算等等。

    .. note::
        现在支持子列表模式。例如，ops.einsum(op1, sublist1, op2, sublist2, ..., sublist_out)。在子列表模式中， `equation` 由子列表推导得到，Python的省略号和介于[0, 52)的整数list组成子列表。每个操作数后面都有一个子列表，并且最后有一个表示输出的子列表。

    参数：
        - **equation** (str) - 基于爱因斯坦求和约定的符号，表示想要执行的操作。符号只能包含字母、逗号、省略号和箭头。字母表示输入Tensor维数，逗号表示单独的Tensor，省略号表示忽略的Tensor维数，箭头的左边表示输入Tensor，右边表示期望输出的维度。
        - **operands** (Tensor) - 用于计算的输入Tensor。Tensor的数据类型必须相同。

    返回：
        Tensor，shape可以根据 `equation` 得到。数据类型和输入Tensor相同。

    异常：
        - **TypeError** - `equation` 无效或者不匹配输入Tensor。
        - **ValueError** - 子列表模式下子列表的数字不介于[0, 52)之间。
