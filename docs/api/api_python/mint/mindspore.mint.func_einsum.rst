mindspore.mint.einsum
=====================

.. py:function:: mindspore.mint.einsum(equation, *operands)

    基于爱因斯坦求和约定（Einsum）符号，沿着指定维度对输入Tensor元素的乘积求和。可以使用这个运算符来执行对角线、求和、转置、矩阵乘法、乘法、内积等运算。

    .. note::
        现在支持子列表模式。例如，mint.einsum(op1, sublist1, op2, sublist2, ..., sublist_out)。在子列表模式中， `equation` 由子列表推导得到，Python的省略号和介于[0, 52)的整数list组成子列表。每个操作数后面都有一个子列表，并且最后有一个表示输出的子列表。
        动态shape、动态rank的输入不支持在 `图模式(mode=mindspore.GRAPH_MODE) <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下执行。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **equation** (str) - 基于爱因斯坦求和约定的符号，表示想要执行的操作。符号只能包含字母（必须在 [a-zA-Z] 中）、逗号、省略号和箭头。字母表示输入Tensor维数，逗号表示单独的Tensor，省略号表示忽略的Tensor维数，箭头的左边表示输入Tensor、右边表示期望输出的维度。如果方程中没有箭头，则方程中仅出现一次的字母将代表一部分的输出的维度，并按字母顺序升序排序。此时输出是先将输入操作数维度根据字母对齐后按元素相乘，然后将不属于输出字母对应的维度进行相加。如果方程中有一个箭头，则代表输出的字母必须在输入字母中出现至少一次，而在输出字母中只能最多出现一次。
        - **operands** (Tensor) - 用于计算的输入Tensor。Tensor的数据类型必须相同。

    返回：
        Tensor，shape可以根据 `equation` 得到。数据类型和输入Tensor相同。

    异常：
        - **TypeError** - `equation` 无效或者不匹配输入Tensor。
        - **ValueError** - 子列表模式下，子列表的数字不介于[0, 52)之间。
