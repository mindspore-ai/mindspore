mindspore.reshard
=================

.. py:function:: mindspore.reshard(tensor, layout)

    指定输入张量的精准排布。其中，传入的layout需要为mindspore.Layout类型，可参考： :class:`mindspore.Layout` 的描述。
    
    - 在图模式下，可以利用此方法设置某个张量的并行切分策略，未设置的会自动通过策略传播方式配置。
    - 在PyNative模式下，可以利用此方法，对某个以图模式进行并行执行的Cell（即PyNative模式下使用了Cell.shard/F.shard的Cell）
      中的张量进行排布指定。

    .. note::
        - 自动并行模式（auto_parallel）下，如果搜索模式（search_mode）不为"sharding_propagation"，则报错。
        - 半自动并行模式（semi_auto_parallel）下，会自动设置为"auto_parallel"且搜索模式自动设置为
          "sharding_propagation"。

    参数：
        - **tensor** (Tensor) - 待设置切分策略的张量。
        - **layout** (Layout) - 指定精准排布的方案，包括描述设备的排布（device_matrix）和设备矩阵的映射别名（alias_name）。

    返回：
        Tensor，与输入的tensor数学等价。

    异常：
        - **TypeError** - Reshard的第一个输入参数需要是Tensor类型，但是当前为 `type(tensor)` 类型。
        - **TypeError** - Reshard只支持输入mindspore.Layout类型作为layout参数，但当前为 `type(layout)` 类型。