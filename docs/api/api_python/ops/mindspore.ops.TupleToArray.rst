mindspore.ops.TupleToArray
===========================

.. py:class:: mindspore.ops.TupleToArray

    将tuple转换为Tensor。

    如果tuple中第一个Number的数据类型为整数，则输出Tensor的数据类型为int。否则，输出Tensor的数据类型为float。

    **输入：**

    - **input_x** (tuple) - Number组成的tuple。这些Number具有相同的类型。仅支持常量值。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    **输出：**

    Tensor。如果输入tuple包含 `N` 个Number，则输出Tensor的shape为(N,)。

    **异常：**

    - **TypeError** - `input_x` 不是tuple。
    - **ValueError** - `input_x` 的长度小于或等于0。