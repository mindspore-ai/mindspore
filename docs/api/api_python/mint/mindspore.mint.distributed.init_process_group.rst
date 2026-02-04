mindspore.mint.distributed.init_process_group
=================================================

.. py:function:: mindspore.mint.distributed.init_process_group(backend="hccl",init_method=None,timeout=None,world_size=-1,rank=-1,store=None,pg_options=None,device_id=None)

    初始化通信服务并创建默认通讯group（group=GlobalComm.WORLD_COMM_GROUP）。

    .. note::
        - 当前接口不支持GPU、CPU版本的MindSpore调用。
        - 在Ascend硬件平台下，这个接口的设置需要在创建Tensor和Parameter之前，以及所有算子和网络的实例化和运行之前。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **backend** (str，可选) - 分布式后端的名称，默认为 ``"hccl"``，且目前只能设置为hccl。
        - **init_method** (str, 可选) - 初始化通讯域时的URL配置，默认为 ``None``。
        - **timeout** (timedelta, 可选) - 设置接口的超时配置，默认为 ``None``。目前，仅在使用 `init_method` 或者 `store` 进行主机端集群网络配置时，才支持该参数。
        - **world_size** (int, 可选) - 配置初始化时全局通讯的卡数，默认为 ``-1``。
        - **rank** (int, 可选) - 设置当前设备卡的卡号，默认为 ``-1``。
        - **store** (Store, 可选) - 存储key/value数据的对象，用于进程间通讯地址、连接信息的交换，默认为 ``None``。目前，仅支持 ``TCPStore`` 类型。
        - **pg_options** (ProcessGroupOptions, 无效参数) - 针对创建的通讯group设置特殊配置策略。该参数为预留参数，当前设置不生效。
        - **device_id** (int, 无效参数) - 设置当前进程使用的NPU卡卡号。该参数为预留参数，当前设置不生效。

    异常：
        - **ValueError** - 参数 `backend` 的值不是 ``"hccl"``。
        - **ValueError** - 参数 `world_size` 的值非 ``-1``，且与实际通讯集群数不一致。
        - **ValueError** - 同时设置参数 `init_method` 和 `store` 。
        - **ValueError** - 当使用 `init_method` 或 `store` 初始化方式时，参数 `world_size` 未被正确设置为正整数。
        - **ValueError** - 当使用 `init_method` 或 `store` 初始化方式时，参数 `rank` 未被正确设置为非负整数。
        - **RuntimeError** - 1）硬件设备类型无效；2）后台服务无效；3）分布式计算初始化失败；4）后端是HCCL的情况下，未设置环境变量 RANK_ID 或 MINDSPORE_HCCL_CONFIG_PATH 的情况下初始化HCCL服务。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
