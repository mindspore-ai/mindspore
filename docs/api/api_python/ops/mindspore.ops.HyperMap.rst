mindspore.ops.HyperMap
=======================

.. py:class:: mindspore.ops.HyperMap(ops=None, reverse=False)

    对输入序列做集合运算。
   
    对序列的每个元素或嵌套序列进行运算。与 `mindspore.ops.Map` 不同，`HyperMap` 能够用于嵌套结构。

    参数：
        - **ops** (Union[MultitypeFuncGraph, None]) -  `ops` 是指定运算操作。如果 `ops` 为None，则运算应该作为 `HyperMap` 实例的第一个入参。默认值为None。
        - **reverse** (bool) -  在某些场景下，需要逆向以提高计算的并行性能，一般情况下，用户可以忽略。`reverse` 用于决定是否逆向执行运算，仅在图模式下支持。默认值为False。

    输入：
        - **args** (Tuple[sequence]) - 如果 `ops` 不是None，则所有入参都应该是具有相同长度的序列，并且序列的每一行都是运算的输入。如果 `ops` 是None，则第一个入参是运算，其余都是输入。

    .. note::
        输入数量等于 `ops` 的输入数量。

    输出：
        序列或嵌套序列，执行函数如 `operation(args[0][i], args[1][i])` 之后输出的序列。

    异常：
        - **TypeError** - 如果 `ops` 既不是 `MultitypeFuncGraph` 也不是None。
        - **TypeError** - 如果 `args` 不是一个tuple。
