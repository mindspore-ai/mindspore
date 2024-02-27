mindspore.nn.PipelineGradReducer
====================================

.. py:class:: mindspore.nn.PipelineGradReducer(parameters, scale_sense=1.0)

    用于流水线并行的梯度聚合。

    参数：
        - **parameters** (list) - 需要更新的参数。
        - **scale_sense** (float) - 梯度的放缩系数，默认为 ``1.0``。

    异常：
        - **RuntimeError** - 如果当前模式不是图模式。
        - **RuntimeError** - 如果并行模式不是半自动并行或者自动并行。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。

