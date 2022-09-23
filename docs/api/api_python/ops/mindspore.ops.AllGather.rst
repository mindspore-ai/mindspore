mindspore.ops.AllGather
========================

.. py:class:: mindspore.ops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)

    在指定的通信组中汇聚Tensor。

    .. note::
        集合中所有进程的Tensor必须具有相同的shape和格式。用户在使用之前需要设置环境变量，运行下面的例子。获取详情请点击官方网站 `MindSpore <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore.ops.html#通信算子>`_ 。

    参数：
        - **group** (str) - 工作的通信组，默认值："GlobalComm.WORLD_COMM_GROUP"（即Ascend平台为"hccl_world_group"，GPU平台为"nccl_world_group" ）。

    输入：
        - **input_x** (Tensor) - AllGather的输入，shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。

    输出：
        Tensor，如果组中的device数量为N，则输出的shape为 :math:`(N, x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - `group` 不是str。
        - **ValueError** - 调用进程的rank id大于本通信组的rank大小。