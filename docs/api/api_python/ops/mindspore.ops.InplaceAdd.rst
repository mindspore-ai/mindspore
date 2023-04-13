mindspore.ops.InplaceAdd
========================

.. py:class:: mindspore.ops.InplaceAdd(indices)

    将 `input_v` 添加到 `x` 的特定行。即计算 `y` = `x`; `y[i,]` += `input_v` 。

    更多参考详见 :func:`mindspore.ops.inplace_add`。

    参数：
        - **indices** (Union[int, tuple]) - 指定将 `input_v` 添加到 `x` 的哪些行。可以为int或Tuple，取值范围[0, len(`x`)]。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为：:math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **input_v** (Tensor) - 添加到 `x` 的Tensor。除第一个维度之外shape必须与 `x` 的shape相同。第一个维度必须与 `indices` 的长度相同。数据类型与 `x` 相同。

    输出：
        Tensor，与 `x` 的shape和数据类型相同。
