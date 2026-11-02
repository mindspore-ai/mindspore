mindspore.ops.ForiLoop
======================

.. py:class:: mindspore.ops.ForiLoop

    在指定范围内执行循环操作。
    ForiLoop算子的执行逻辑可以近似表示为如下代码：

    .. code-block:: python

        def ForiLoop(lower, upper, loop_func, init_val):
            for i in range(lower, upper):
                init_val = loop_func(i, init_val)
            return init_val

    当前ForiLoop算子存在以下语法限制：

    - 暂不支持 `loop_func` 为副作用函数，例如对Parameter、全局变量的修改等操作。
    - 暂不支持 `loop_func` 的返回值与初始值 `init_val` 的类型或形状不同。
    - 暂不支持负数或自定义增量。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **lower** (Union[int, Tensor]) - 循环的起始索引值。
        - **upper** (Union[int, Tensor]) - 循环的结束索引值。
        - **loop_func** (Function) - 循环体函数。
        - **init_val** (Union[Tensor, number, str, bool, list, tuple, dict]) - 循环的初始值。
        - **unroll** (bool, 可选) - 是否在编译阶段展开，只在循环次数确定的情况下有效。默认值： ``True`` 。

    输出：
        Union[Tensor, number, str, bool, list, tuple, dict]，循环的最终结果，和 `init_val` 的类型和形状相同。

    异常：
        - **TypeError** - `lower` 不是一个整数或者Tensor。
        - **TypeError** - `upper` 不是一个整数或者Tensor。
        - **TypeError** - `loop_func` 不是一个函数。
        - **ValueError** - `loop_func` 不能接受索引值和 `init_val` 作为参数，或者输出值和 `init_val` 的类型或形状不同。
