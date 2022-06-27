mindspore.ops.ReduceStd
=======================

.. py:class:: mindspore.ops.ReduceStd(axis=(), unbiased=True, keep_dims=False)

    默认情况下，输出Tensor各维度上的标准差与均值，也可以对指定维度求标准差与均值。如果 `axis` 是维度列表，则减少对应的维度。

    通过指定 `keep_dims` 参数，来控制输出和输入的维度是否相同。

    更多参考详见 :func:`mindspore.ops.std`。