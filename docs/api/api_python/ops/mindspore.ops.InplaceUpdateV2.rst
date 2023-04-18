mindspore.ops.InplaceUpdateV2
=============================

.. py:class:: mindspore.ops.InplaceUpdateV2()

    根据 `indices`，将 `x` 中的某些值更新为 `v`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.inplace_update`。

    输入：
        - **x** (Tensor) - 待更新的Tensor。数据类型支持float16、float32或int32。
        - **indices** (Union[int, tuple, Tensor]) - 指定将 `x` 的哪些行更新为  `v` 。可以为int或Tuple或Tensor，取值范围[0, len(`x`))。
        - **v** (Tensor) - 除第一个维度之外shape必须与 `x` 的shape相同。第一个维度必须与 `indices` 的长度相同。数据类型与 `x` 相同。

    输出：
        Tensor，与 `x` 的shape和数据类型相同。
