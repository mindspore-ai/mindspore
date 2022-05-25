mindspore.ops.InplaceAdd
========================

.. py:class:: mindspore.ops.InplaceAdd(indices)

    将 `input_v` 添加到 `x` 的特定行。即计算 `y` = `x`; `y[i,]` += `input_v` 。

    **参数：**

    - **indices** (Union[int, tuple]) - 指定将 `input_v` 添加到 `x` 的哪些行。可以为int或Tuple，取值范围[0, len(`x`)]。

    **输入：**

    - **x** (Tensor) - 数据类型支持float16、float32或int32。shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其轶应小于8。
    - **input_v** (Tensor) - 除第一个维度之外shape必须与 `x` 的shape相同。第一个维度必须与 `indices` 的长度相同。数据类型与 `x` 相同。

    **输出：**

    Tensor, 与 `x` 的shape和数据类型相同。

    **异常：**

    - **TypeError** - `indices` 不是int或tuple。
    - **TypeError** - `indices` 为Tuple，而其包含的某一元素非int类型。
    - **ValueError** - `x` 的shape尺寸与 `input_v` 的shape尺寸不同。

