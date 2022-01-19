mindspore.ops.Fill
==================

.. py:class:: mindspore.ops.Fill()

    创建一个指定shape的Tensor，并用指定值填充。

    **输入：**

    - **type** (mindspore.dtype) - 指定输出Tensor的数据类型。数据类型只支持 `bool_ <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_ 和 `number <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_ 。
    - **shape** (tuple[int]) - 指定输出Tensor的shape。
    - **value** (Union(number.Number, bool)) - 用来填充输出Tensor的值。

    **输出：**

    Tensor。

    **异常：**

    **TypeError** - `shape` 不是元组。
