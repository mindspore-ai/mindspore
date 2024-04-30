mindspore.Tensor.register_hook
==============================

.. py:method:: mindspore.Tensor.register_hook(hook_fn)

    设置Tensor对象的反向hook函数。

    .. note::
        - `register_hook(hook_fn)` 在图模式下，或者在PyNative模式下使用 `jit` 装饰器功能时不起作用。
        - hook_fn必须有如下代码定义：`grad` 是反向传递给Tensor对象的梯度。 用户可以在hook_fn中打印梯度数据或者返回新的输出梯度。
        - hook_fn返回新的梯度输出，不能不设置返回值：hook_fn(grad) -> New grad_output。

    参数：
        - **hook_fn** (function) - 捕获Tensor反向传播时的梯度，并输出或更改该梯度的 `hook_fn` 函数。

    返回：
        返回与该hook_fn函数对应的handle对象。可通过调用handle.remove()来删除添加的hook_fn函数。

    异常：
        - **TypeError** - 如果 `hook_fn` 不是Python函数。
