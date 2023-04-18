mindspore.ops.GatherNd
=======================

.. py:class:: mindspore.ops.GatherNd

    根据索引获取输入Tensor指定位置上的元素。

    更多参考详见 :func:`mindspore.ops.gather_nd`。

    输入：
        - **input_x** (Tensor) - 目标Tensor。
        - **indices** (Tensor) - 索引Tensor，其数据类型为int32或int64。

    输出：
        Tensor，数据类型与 `input_x` 相同，shape为 `indices_shape[:-1] + x_shape[indices_shape[-1]:]` 。
