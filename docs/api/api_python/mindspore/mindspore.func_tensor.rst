mindspore.tensor
================

.. py:function:: mindspore.tensor(input_data=None, dtype=None, shape=None, init=None, internal=False, const_arg=False)

    此接口用于在Cell.construct()或者@jit装饰的函数内，创建一个新的Tensor对象。

    在图模式下，MindSpore可以在运行时依据 `dtype` 参数来动态创建新Tensor。

    详情请参考教程 `创建和使用Tensor <https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#mindspore自定义数据类型>`_ 。

    有别于Tensor类，其与Tensor类的区别为内部增加了 `Annotation <https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html#annotation-type>`_ 指示当前创建的Tensor的类型，与Tensor类相比能够防止AnyType的产生。

    参数和返回值与Tensor类完全一致。另参考：:class:`mindspore.Tensor`。


