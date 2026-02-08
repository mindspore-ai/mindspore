mindspore.saved_tensors_hooks
=======================================

.. py:class:: mindspore.saved_tensors_hooks(pack_hook, unpack_hook)

    一个上下文管理器，用于自定义保存张量（Saved Tensor）的打包（pack）和解包（unpack）方式。

    在前向计算中，某些张量会被保存，以供反向传播时使用。通过使用该上下文，用户可以指定：

    - 在保存前如何处理这些张量（打包阶段）。
    - 在反向访问时如何恢复这些张量（解包阶段）。

    打包和解包函数应符合以下签名：

    - ``pack_hook(tensor: Tensor) -> Any``：
      接收一个张量，并返回任意对象，用于表示该张量在存储阶段的形式。

    - ``unpack_hook(packed: Any) -> Tensor``：
      接收上述返回值，并恢复出对应的张量。

    .. note::
        当前该上下文管理器在Graph模式与Jit模式下不支持。

    .. warning::
        - 为防止未定义行为，对传入 `pack_hook` 的原始张量进行原地修改会抛出异常。
        - 为避免产生循环引用， `pack_hook` 返回的对象不能直接持有原始张量的引用。

    参数：
        - **pack_hook** (Callable) - 定义前向计算保存张量时的处理方法。
        - **unpack_hook** (Callable) - 定义反向计算恢复张量时的处理方法。
