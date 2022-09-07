mindspore.ops.Map
==================

.. py:class:: mindspore.ops.Map(ops=None, reverse=False)

    Map将对输入序列应用设置的函数操作。

    此操作将应用到输入序列的每个元素。

    参数：
        - **ops** (Union[MultitypeFuncGraph, None]) - `ops` 是要应用的操作。如果 `ops` 为None，则操作应放在实例的第一个输入中。默认值：None。
        - **reverse** (bool) - 在某些场景中，优化器需要反转以提高并行性能，一般用户可忽略。 `reverse` 表示是否为反向应用操作的标志。仅支持图模式。默认值：False。

    输入：
        - **args** (Tuple[sequence]) - 如果 `ops` 不是None，则所有输入的序列和序列的每一行都应该是相同长度。例如，如果 `args` 的长度为2，那么每个序列 `(args[0][i],args[1][i])` 长度的 `i` 将作为操作的输入。如果 `ops` 为None，则第一个输入是操作，另一个输入是输入。

    输出：
        序列，进行函数操作后的输出序列。例如 `operation(args[0][i], args[1][i])` 。
