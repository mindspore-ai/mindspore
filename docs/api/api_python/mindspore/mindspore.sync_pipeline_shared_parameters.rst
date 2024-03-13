mindspore.sync_pipeline_shared_parameters.rst
================================================

.. py:function:: mindspore.sync_pipeline_shared_parameters(net)

    在流水线并行场景下，部分参数可能会被不同的stage之间共享。例如`embedding table`被VocabEmbedding和`LMHead`两层共享，这两层通常会被切分到不同的stage上。
    在流水线并行推理时，有必要`embedding table`变更后在stage之间进行权重同步。

    .. note::
        网络需要先编译，再执行stage之间权重同步。

    参数：
        - **net** (nn.Cell) - 推理网络
