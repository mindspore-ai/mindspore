mindspore.mutable
==================

.. py:function:: mindspore.mutable(input_data, dynamic_len=False)

    设置一个常量值为可变的。

    当前除了Tensor，所有顶层网络的输入，例如标量、tuple、list和dict，都被当做常量值。常量值是不能求导的，而且在编译优化阶段会被常量折叠掉。
    另外，当网络的输入是tuple[Tensor], list[Tensor]或Dict[Tensor]时，即使里面Tensor的shape和dtype没有发生变化，在多次调用同一个网络的时候，这个网络每次都会被重新编译，这是因为这些类型的输入被当做常量值处理了。

    为解决以上的问题，我们提供了 `mutable` 接口去设置网络的常量输入为"可变的"。一个"可变的"输入意味着这个输入成为了像Tensor一样的变量，最重要的是，我们可以对其进行求导了。

    当 `input_data` 是tuple或者list并且 `dynamic_len` 是False的情况下，`mutable` 的返回值是一个固定长度的tuple或者list，且其中的每一个元素都是可变的。当 `dyanmic_len` 被设置为True的时候，返回的tuple或者list长度是动态的。

    如果一个动态长度的tuple或者list被作为网络的输入并且这个网络被重复调用，且每一次的输入的tuple或者list长度都不一致，这个网络也不需要被重新编译。

    参数：
        - **input_data** (Union[int, float, Tensor, tuple, list, dict]) - 要设置为可变的输入数据。如果 `input_data` 是list，tuple或者dict， 其内部元素的类型也需要是这些有效类型中的一个。
        - **dynamic_len** (bool) - 是否要将整个序列设置为动态长度的。在图编译内，如果 `dynamic_len` 被设置为True， 那么 `input_data` 必须为tuple或者list， 并且其中的元素必须有相同的类型以及形状。默认值：False。

    .. warning::
        - 这是一个实验特性，未来有可能被修改或删除。
        - 当前该接口只在图模式下生效。

    返回：
        状态设置为可变的原输入数据。

    异常：
        - **TypeError** - 如果 `input_data` 不是int、float、tuple[Tensor]、list[Tensor]或dict[Tensor]的其中一种类型或者不是它们的嵌套结构。
        - **TypeError** - 如果 `dynamic_len` 被设置为True并且 `input_data` 不是tuple或者list。
        - **ValueError** - 如果 `dynamic_len` 被设置为True， `input_data` 是tuple或者list的情况下，其中的元素的形状或者类型不一致。
