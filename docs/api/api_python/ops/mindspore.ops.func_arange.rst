mindspore.ops.arange
===============================

.. py:function:: mindspore.ops.arange(start=0, stop=None, step=1, rtype=None)

    根据给定的范围返回指定均匀间隔的数据。

    **参数：**

    - **start** (Union[int, float]) - 指定范围的起始值，范围包含该值。类型为int或float。
    - **stop** (Union[int, float]) - 指定范围的结束值，范围不包含该值。类型为int或float。
    - **step** (Union[int, float]) - 指定取值的间隔。类型为int或float。
    - **rtype** (Union[mindspore.dtype，str]) - 指定返回数据的类型，如果不指定，则会根据 `start` 、 `stop` 、 `step` 的值推断类型。

    **返回：**

    Tensor，值是均匀间隔的数据，类型为给定或推断的结果。

    **异常：**

    - **TypeError** -  `start` 、 `stop` 、 `step` 的类型不是int或float。
    - **ValueError** - `start` 的值大于等于 `stop` 。