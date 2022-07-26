mindspore.ops.InplaceUpdate
===========================

.. py:class:: mindspore.ops.InplaceUpdate(indices)

    将 `x` 的特定行更新为 `v` 。

    参数：
        - **indices** (Union[int, tuple]) - 指定将 `x` 的哪些行更新为  `v` 。可以为int或Tuple，取值范围[0, len(`x`))。

    输入：
        - **x** (Tensor) - 待更新的Tensor。数据类型支持float16、float32或int32。
        - **v** (Tensor) - 除第一个维度之外shape必须与 `x` 的shape相同。第一个维度必须与 `indices` 的长度相同。数据类型与 `x` 相同。

    输出：
        Tensor，与 `x` 的shape和数据类型相同。

    异常：
        - **TypeError** - `indices` 不是int或Tuple。
        - **TypeError** - `indices` 为Tuple，而其包含的某一元素非int类型。

