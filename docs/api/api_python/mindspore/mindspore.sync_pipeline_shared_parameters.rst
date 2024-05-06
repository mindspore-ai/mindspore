mindspore.sync_pipeline_shared_parameters
================================================

.. py:function:: mindspore.sync_pipeline_shared_parameters(net)

    在流水线并行场景下，部分参数可能会被不同的stage之间共享。例如 `embedding table` 被 `VocabEmbedding` 和 `LMHead` 两层共享，这两层通常会被切分到不同的stage上。
    在流水线并行推理时，有必要 `embedding table` 变更后在stage之间进行权重同步。

    .. note::
        网络需要先编译，再执行stage之间权重同步。

    参数：
        - **net** (nn.Cell) - 推理网络。

    支持平台：
        ``Ascend``

    样例：

    .. note::
        运行以下样例之前，需要配置好通信环境变量。

        针对Ascend设备，用户需要编写动态组网启动脚本，详见 `动态组网启动 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dynamic_cluster.html>`_ 。
