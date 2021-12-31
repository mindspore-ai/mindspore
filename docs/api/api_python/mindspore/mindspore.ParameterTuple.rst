mindspore.ParameterTuple
========================

.. py:class:: mindspore.ParameterTuple(iterable)

    参数元组的类。

    .. note::
        该类把网络参数存储到参数元组集合中。
    
    .. py:method:: clone(prefix, init='same')

        按元素克隆 `ParameterTuple` 中的数值，以生成新的 `ParameterTuple` 。

        **参数：**

        - **prefix** (str) - 参数的命名空间。
        - **init** (Union[Tensor, str, numbers.Number]) - 初始化参数的shape和dtype。 `init` 的定义与 `Parameter` API中的定义相同。默认值：'same'。

        **返回：**

        新的参数元组。