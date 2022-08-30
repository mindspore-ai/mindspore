mindspore.ops.tuple_to_array
==============================

.. py:function:: mindspore.ops.tuple_to_array(input_x)

    将tuple转换为Tensor。

    如果tuple中第一个数据类型为int，则输出Tensor的数据类型为int。否则，输出Tensor的数据类型为float。

    参数：
        - **input_x** (tuple) - 数值型组成的tuple。其元素具有相同的类型。仅支持常量值。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor。如果输入tuple包含 `N` 个数值型元素，则输出Tensor的shape为(N,)。

    异常：
        - **TypeError** - `input_x` 不是tuple。
        - **ValueError** - `input_x` 的长度小于或等于0。