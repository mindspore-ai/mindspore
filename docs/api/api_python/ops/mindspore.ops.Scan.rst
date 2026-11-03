mindspore.ops.Scan
====================

.. py:class:: mindspore.ops.Scan

    将一个函数循环作用于一个数组，且对当前元素的处理依赖上一个元素的执行结果。
    Scan算子的执行逻辑可以近似表示为如下代码：

    .. code-block:: python

        def Scan(loop_func, init, xs, length=None):
            if xs is None:
                xs = [None] * length
            carry = init
            ys = []
            for x in xs:
                carry, y = loop_func(carry, x)
                ys.append(y)
            return carry, ys

    当前Scan算子存在以下语法限制：

    - 暂不支持 `loop_func` 为副作用函数，如：对Parameter、全局变量的修改等操作。
    - 暂不支持 `loop_func` 的返回值的第一个元素与初始值 `init` 的类型或形状不同。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **loop_func** (Function) - 循环体函数。
        - **init** (Union[Tensor, number, str, bool, list, tuple, dict]) - 循环的初始值。
        - **xs** (Union[tuple, list, None]) - 用于执行循环扫描的数组。
        - **length** (Union[int, None], 可选) - 数组xs的长度，默认值： ``None`` 。
        - **unroll** (bool, 可选) - 是否在编译阶段进行循环展开，默认值： ``True`` 。

    输出：
        Tuple(Union[Tensor, number, str, bool, list, tuple, dict], list)，
        由两个元素组成的tuple，第一个元素为循环的最终结果，和 `init` 参数保持一样的类型和形状；
        第二个元素是一个列表，包含每次循环的执行结果。

    异常：
        - **TypeError** - `loop_func` 不是一个函数。
        - **TypeError** - `xs` 不是一个tuple、一个list或者None。
        - **TypeError** - `length` 不是一个整数或者None。
        - **TypeError** - `unroll` 不是一个布尔值。
        - **ValueError** - `loop_func` 不能接受 `init` 以及 `xs` 的元素作为参数。
        - **ValueError** - `loop_func` 的返回值不是一个包含两个元素的tuple，或者tuple的第一个元素与 `init` 的类型或形状不同。
