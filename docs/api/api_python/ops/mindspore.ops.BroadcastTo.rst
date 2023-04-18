mindspore.ops.BroadcastTo
==========================

.. py:class:: mindspore.ops.BroadcastTo(shape)

    将输入shape广播到目标shape。

    更多细节请参考 :func:`mindspore.ops.broadcast_to` 。

    参数：
        - **shape** (tuple) - 指定广播的目标shape。

    输入：
        - **input_x** (Tensor) - BroadcastTo输入，是任意维度的Tensor。

    输出：
        Tensor，与目标 `shape` 相同，数据类型与 `input_x` 相同。
