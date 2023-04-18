mindspore.ops.GatherD
=======================

.. py:class:: mindspore.ops.GatherD

    获取指定轴的元素。

    更多参考详见 :func:`mindspore.ops.gather_elements`。

    输入：
        - **x** (Tensor) - 输入Tensor。
        - **dim** (int) - 获取元素的轴。数据类型为int32或int64。只能是常量值。
        - **index** (Tensor) - 获取收集元素的索引。支持的数据类型包括：int32，int64。每个索引元素的取值范围为[-x_rank[dim], x_rank[dim])。

    输出：
         Tensor，数据类型与 `x` 相同。
