mindspore.recompute
===================

.. py:function:: mindspore.recompute(block, *args, use_reentrant=True, output_recompute=False, **kwargs)

    该函数用于减少显存的使用，当运行选定的模块时，不再保存其中的前向计算产生的激活值，我们将在反向传播时，重新计算前向的激活值。

    .. note::
        重计算函数只支持继承自Cell对象的模块。

    参数：
        - **block** (Cell) - 需要重计算的网络模块。
        - **args** (tuple) - 指需要重计算的网络模块的前向输入。

    关键字参数：
        - **use_reentrant** (bool, 可选) - 该参数只在PyNative模式下有效。若设置为 ``True``，将通过自定义反向传播函数实现重计算，该方式不支持List/Tuple等复杂类型的求导；若设置为 ``False``，将使用 :class:`mindspore.saved_tensors_hooks` 实现重计算，该方式支持对复杂类型内部张量的求导。默认值： ``True`` 。
        - **output_recompute** (bool, 可选) - 该参数只在PyNative模式下有效。若设置 ``True``，默认将使用 :class:`mindspore.saved_tensors_hooks` 实现重计算。该模块的输出不会被后续需要求导的算子缓存。当存在两个相邻cell均需重计算时（其中一个cell的输出作为另一个cell的输入），这两个cell的重计算将被融合。在此情况下，第一个cell的输出激活值将不会被保存。默认值： ``False`` 。
        - **\*\*kwargs** - 其他参数。

    返回：
        同 `block` 的返回类型相同。

    异常：
        - **TypeError** - 如果 `block` 不是Cell对象。

