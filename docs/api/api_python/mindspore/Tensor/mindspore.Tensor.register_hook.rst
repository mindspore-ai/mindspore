mindspore.Tensor.register_hook
==============================

.. py:method:: mindspore.Tensor.register_hook(hook)

    设置Tensor对象的反向hook函数。

    .. note::
        - `hook` 必须有如下代码定义： `grad` 是反向传递给 `Tensor` 对象的梯度。用户可以在 `hook` 中打印梯度数据或者返回新的输出梯度。
        - `hook` 返回新的梯度输出，必须设置返回值： `hook(grad) -> New grad_output`。
        - 静态图模式下需满足如下约束：

          - `hook` 同样需满足静态图模式下的语法约束。
          - 不支持在图内对 `hook` 进行删除。
          - 不支持在 `Tensor` 被使用之后再次对 `Tensor` 注册 `hook`。
          - 不支持在图内对 `Tensor` 注册多个 `hook`。
          - 图内对 `Tensor` 注册 `hook` 将返回 `Tensor` 本身。

    参数：
        - **hook** (function) - 捕获 `Tensor` 反向传播时的梯度，并输出或更改该梯度的 `hook` 函数。

    返回：
        返回与该 `hook` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook` 函数。

    异常：
        - **TypeError** - 如果 `hook` 不是Python函数。
