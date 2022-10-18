mindspore.jit
=============

.. py:function:: mindspore.jit(fn=None, input_signature=None, hash_args=None, jit_config=None)

    将Python函数编译为一张可调用的MindSpore图。

    MindSpore可以在运行时对图进行优化。

    参数：
        - **fn** (Function) - 要编译成图的Python函数。默认值：None。
        - **input_signature** (Tensor) - 用于表示输入参数的Tensor。Tensor的shape和dtype将作为函数的输入shape和dtype。默认值：None。
        - **hash_args** (Union[Object, List or Tuple of Objects]) - `fn` 里面用到的自由变量，比如外部函数或类对象，再次调用时若 `hash_args` 出现变化会触发重新编译。默认值：None。
        - **jit_config** (JitConfig) - 编译时所使用的JitConfig配置项，详细可参考 :class:`mindspore.JitConfig`。默认值：None。

    .. note::
        - 如果指定了 `input_signature` ，则 `fn` 的每个输入都必须是Tensor。并且 `fn` 的输入参数将不会接受 `**kwargs` 参数。

    返回：
        函数，如果 `fn` 不是None，则返回一个已经将输入 `fn` 编译成图的可执行函数；如果 `fn` 为None，则返回一个装饰器。当这个装饰器使用单个 `fn` 参数进行调用时，等价于 `fn` 不是None的场景。
