mindspore.ops.HookBackward
===========================

.. py:class:: mindspore.ops.HookBackward(hook_fn, cell_id="")

    用来导出中间变量中的梯度。请注意，此函数仅在PyNative模式下支持。

    .. note::
        钩子函数必须定义为 `hook_fn(grad) -> new gradient or None` ，其中'grad'是传递给Primitive的梯度。可以通过返回新的梯度并传递到下一个Primitive来修改'grad'。钩子函数和InsertGradientOf的回调的区别在于，钩子函数是在python环境中执行的，而回调将被解析并添加到图中。

    参数：
        - **hook_fn** (Function) - Python函数。钩子函数。
        - **cell_id** (str，可选) - 用于标识钩子注册的函数是否实际注册在指定的cell对象上。例如，:class:`mindspore.nn.Conv2d` 是一个cell对象。默认值： ``""`` ，此情况下系统将自动注册 `cell_id` 的值。 此参数目前不支持自定义。

    输入：
        - **input** (Tensor) - 需要导出的变量的梯度。

    输出：
        - **output** (Tensor) - 直接返回 `input` 。 `HookBackward` 不影响前向结果。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `hook_fn` 不是Python的函数。
