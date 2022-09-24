mindspore.mutable
==================

.. py:function:: mindspore.mutable(input_data)

    设置一个常量值为可变的。

    当前除了Tensor，所有顶层网络的输入，例如标量、tuple、list和dict，都被当做常量值。常量值是不能求导的，而且在编译优化阶段会被常量折叠掉。
    另外，当网络的输入是tuple[Tensor], list[Tensor]或Dict[Tensor]时，即使里面Tensor的shape和dtype没有发生变化，在多次调用同一个网络的时候，这个网络每次都会被重新编译，这是因为这些类型的输入被当做常量值处理了。

    为解决以上的问题，我们提供了 `mutable` 接口去设置网络的常量输入为"可变的"。一个"可变的"输入意味着这个输入成为了像Tensor一样的变量。最重要的是，我们可以对其进行求导了。

    参数：
        - **input_data** (Union[Tensor, tuple[Tensor], list[Tensor], dict[Tensor]]) - 要设置为可变的输入数据。

    .. warning::
        - 这是一个实验特性，未来有可能被修改或删除。
        - 目前运行时暂时不支持处理标量数据流，所以我们目前只支持Tensor、tuple[Tensor]、list[Tensor]或dict[Tensor]作为输入，主要解决重复编译的问题。
        - 当前该接口只在图模式下生效。

    返回：
        状态设置为可变的原输入数据。

    异常：
        - **TypeError** - 如果 `input_data` 不是Tensor、tuple[Tensor]、list[Tensor]或dict[Tensor]的其中一种类型或者不是它们的嵌套结构。
