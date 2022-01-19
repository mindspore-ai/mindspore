mindspore.ms_function
=====================

.. py:class:: mindspore.ms_function(fn=None, obj=None, input_signature=None)

    将Python函数编译为一张可调用的MindSpore图。

    MindSpore可以在运行时对图进行优化。

    **参数：**

    - **fn**  (Function) - 要编译成图的Python函数。默认值：None。
    - **obj**  (Object) - 用于区分编译后函数的Python对象。默认值：None。
    - **input_signature** (Tensor) - 用于表示输入参数的Tensor。Tensor的shape和dtype将作为函数的输入shape和dtype。默认值：None。

    .. note::
        - 如果指定了 `input_signature` ，则 `fn` 的每个输入都必须是Tensor。并且 `fn` 的输入参数将不会接受 `**kwargs` 参数。

    **返回：**

    函数，如果 `fn` 不是None，则返回一个已经将输入 `fn` 编译成图的可执行函数；如果 `fn` 为None，则返回一个装饰器。当这个装饰器使用单个 `fn` 参数进行调用时，等价于 `fn` 不是None的场景。

    **异常：**

    - **TypeError** – 如果指定了 `input_signature`，但是实际输入的shape和dtype与 `input_signature` 的不相同。
