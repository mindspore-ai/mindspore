mindspore.ops.fill
==================

.. py:function:: mindspore.ops.fill(type, shape, value)

    创建一个指定shape的Tensor，并用指定值填充。

    参数：
        - **type** (mindspore.dtype) - 指定输出Tensor的数据类型。数据类型只支持 `bool_ <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 和 `number <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。
        - **shape** (tuple[int]) - 指定输出Tensor的shape。
        - **value** (Union(number.Number, bool)) - 用来填充输出Tensor的值。

    返回：
        Tensor。

    异常：
        - **TypeError** - `shape` 不是元组。
