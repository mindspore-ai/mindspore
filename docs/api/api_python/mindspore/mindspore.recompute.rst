mindspore.recompute
===================

.. py:function:: mindspore.recompute(block, *args, **kwargs)

    该函数用于减少显存的使用，当运行选定的模块时，不再保存其中的前向计算的产生的激活值，我们将在反向传播时，重新计算前向的激活值。

    .. note::
        - 重计算函数只支持继承自Cell对象的模块，
        - 该函数当前只支持PyNative模式，在图模式下，可以尝试使用Cell.recompute()接口，
        - 当使用重计算函数时，传入的网络模块不能使用jit装饰器。

    参数：
        - **block** (Cell) - 需要重计算的网络模块。
        - **args** (tuple) - 指需要重计算的网络模块的前向输入。
        - **kwargs** (dict) - 可选输入。

    返回：
        同block的返回类型相同。

    异常：
        - **TypeError** - 如果 `block` 不是Cell对象。
        - **AssertionError** - 如果执行模式不是PyNative模式。

