mindspore.ops.AlltoAll
======================

.. py:class:: mindspore.ops.AlltoAll(split_count, split_dim, concat_dim, group=GlobalComm.WORLD_COMM_GROUP)

    AlltoAll是一个集合通信函数。

    AlltoAll将输入数据在特定的维度切分成特定的块数（blocks），并按顺序发送给其他rank。一般有两个阶段：

    - 分发阶段：在每个进程上， 操作数沿着 `split_dim` 拆分为 `split_count` 个块（blocks），且分发到指定的rank上，例如，第i块被发送到第i个rank上。
    - 聚合阶段：每个rank沿着 `concat_dimension` 拼接接收到的数据。

    .. note::
        聚合阶段，所有进程中的Tensor必须具有相同的shape和格式。用户在运行以下示例之前需要预置环境变量，请在 `MindSpore <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html#通信算子>`_ 官网查看详细信息。

        要求全连接配网方式，每台设备具有相同的vlan id，ip和mask在同一子网，请查看 `详细信息 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#注意事项>`_ 。

    参数：
        - **split_count** (int) - 在每个进程上，将块（blocks）拆分为 `split_count` 个。
        - **split_dim** (int) - 在每个进程上，沿着 `split_dim` 维度进行拆分。
        - **concat_dim** (int) - 在每个进程上，沿着 `concat_dimension` 拼接接收到的块（blocks）。
        - **group** (str) - AlltoAll的通信域。默认值："GlobalComm.WORLD_COMM_GROUP"。

    异常：
        - **TypeError** - 如果 `group` 不是字符串。
