mindspore.tensor
================

.. py:function:: mindspore.tensor(input_data=None, dtype=None, shape=None, init=None, internal=False, const_arg=False)

    此接口用于在Cell.construct()或者@jit装饰的函数内，创建一个新的Tensor对象。

    有别于Tensor类，在图模式下，MindSpore可以在运行时依据 `dtype` 参数来动态创建新Tensor。

    参数和返回值与Tensor类完全一致。另参考：:class:`mindspore.Tensor`。

