mindspore.ops.gather_elements
=============================

.. py:function:: mindspore.ops.gather_elements(input, dim, index)

    根据指定维度和索引获取元素。

    .. note::
        `input` 与 `index` 维度大小一致，当 `input` 中的 `axis != dim` 时 ， `index.shape[axis] <= input.shape[axis]` 。

    .. warning::
        在Ascend后端，以下场景将导致不可预测的行为：

        - 正向执行流程中，当 `index` 的取值不在范围 `[-input.shape[dim], input.shape[dim])` 内；
        - 反向执行流程中，当 `index` 的取值不在范围 `[0, input.shape[dim])` 内。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 指定维度。
        - **index** (Tensor) - 指定索引。

    返回：
        Tensor
