mindspore.parallel.nn.PipelineGradReducer
============================================================================

.. py:class:: mindspore.parallel.nn.PipelineGradReducer(parameters, scale_sense=1.0, opt_shard=None)

    函数式训练场景下，实现流水线并行的梯度归约及累加。

    参数：
        - **parameters** (list) - 将进行pp并行的网络参数。
        - **scale_sense** (float，可选) - 梯度的尺度感知。默认值： ``1.0``。
        - **opt_shard** (bool，可选) - 如果使用优化器，需要配置为True。默认值： ``None``。

    异常：
        - **RuntimeError** - `mode` 不是图模式。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在多设备上运行。
