mindspore.jit
=============

.. py:function:: mindspore.jit(fn=None, mode="PSJit", input_signature=None, hash_args=None, jit_config=None, compile_once=False)

    将Python函数编译为一张可调用的MindSpore图。

    MindSpore可以在运行时对图进行优化。

    参数：
        - **fn** (Function) - 要编译成图的Python函数。默认值： ``None`` 。
        - **mode** (str) - 使用jit的类型，可选值有 ``"PSJit"`` 和 ``"PIJit"`` 。默认值： ``"PSJit"``。

          - `PSJit <https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html>`_ ：解析python的ast以构建静态图。
          - `PIJit <https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html>`_ ：在运行时解析python字节码以构建静态图。

        - **input_signature** (Union[Tuple, List, Dict, Tensor]) - 输入的Tensor是用于描述输入参数的。Tensor的shape和dtype将被配置到函数中去。如果指定了 `input_signature`，则 `fn` 的输入参数不接受 `**kwargs` 类型，并且实际输入的shape和dtype需要与 `input_signature` 相匹配。否则，将会抛出TypeError异常。 `input_signature` 有两种模式：

          - 全量配置模式：参数为Tuple、List或者Tensor，它们将被用作图编译时的完整编译参数。
          - 增量配置模式：参数为Dict，它将被配置到图的部分输入上，替换图编译对应位置上的参数。

          默认值： ``None`` 。
        - **hash_args** (Union[Object, List or Tuple of Objects]) - `fn` 里面用到的自由变量，比如外部函数或类对象，再次调用时若 `hash_args` 出现变化会触发重新编译。默认值： ``None`` 。
        - **jit_config** (JitConfig) - 编译时所使用的JitConfig配置项，详细可参考 :class:`mindspore.JitConfig`。默认值： ``None`` 。
        - **compile_once** (bool) - ``True``: 函数多次重新创建只编译一次，如果函数里面的自由变量有变化，设置True是有正确性风险； ``False``: 函数重新创建会触发重新编译。默认值： ``False`` 。

    .. note::
        - 如果指定了 `input_signature` ，则 `fn` 的每个输入都必须是Tensor。并且 `fn` 的输入参数将不会接受 `**kwargs` 参数。

    返回：
        函数，如果 `fn` 不是None，则返回一个已经将输入 `fn` 编译成图的可执行函数；如果 `fn` 为None，则返回一个装饰器。当这个装饰器使用单个 `fn` 参数进行调用时，等价于 `fn` 不是None的场景。
