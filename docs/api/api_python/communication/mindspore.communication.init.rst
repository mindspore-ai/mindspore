mindspore.communication.init
============================

.. py:function:: mindspore.communication.init(backend_name=None)

    初始化通信服务需要的分布式后端，例如 `HCCL` 、 `NCCL` 或 `MCCL` 服务。通常在分布式并行场景下使用，并在使用通信服务前设置。

    .. note::
        - HCCL的全称是华为集合通信库（Huawei Collective Communication Library）。
        - NCCL的全称是英伟达集合通信库（NVIDIA Collective Communication Library）。
        - MCCL的全称是MindSpore集合通信库（MindSpore Collective Communication Library）。
        - 在Ascend硬件平台下，``init()`` 接口的设置需要在创建Tensor和Parameter之前，以及所有算子和网络的实例化和运行之前。

    参数：
        - **backend_name** (str, 可选) - 分布式后端的名称，支持 ``"hccl"``、``"nccl"`` 或 ``"mccl"``。在Ascend硬件平台下，应使用 ``"hccl"``；在GPU硬件平台下，应使用 ``"nccl"``；在CPU硬件平台下，应使用 ``"mccl"``。如果未设置则根据硬件平台类型（device_target）自动进行推断。默认值：``None``。

    异常：
        - **TypeError** - 参数 `backend_name` 不是字符串。
        - **RuntimeError** - 硬件设备类型无效、后端服务无效，或分布式计算初始化失败。
        - **RuntimeError** - 后端是HCCL的情况下，未设置环境变量 `RANK_ID` 或 `MINDSPORE_HCCL_CONFIG_PATH` 时初始化HCCL服务。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
