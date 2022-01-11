mindspore.communication
========================
集合通信接口。

注意，集合通信接口需要预先设置环境变量。对于Ascend，用户需要配置rank_table，设置rank_id和device_id，相关教程可参考：
<https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_ascend.html>`_。
对于GPU，用户需要预先配置host_file以及mpi，相关教程参考：
<https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_gpu.html>`_。

目前尚不支持CPU。

.. py:class:: mindspore.communication.GlobalComm

    GlobalComm 是一个储存通信信息的全局类。 成员包含：BACKEND、WORLD_COMM_GROUP。

    - BACKEND：使用的通信库，HCCL或者NCCL。
    - WORLD_COMM_GROUP：全局通信域。

    **样例：**

    >>> from mindspore.context import set_context
    >>> set_context(device_target="Ascend")
    >>> init()
    >>> GlobalComm.BACKEND

.. py:method:: mindspore.communication.init(backend_name=None)

    初始化通信服务需要的分布式后端，例如‘HCCL’或‘NCCL’服务。

    .. note::HCCL的全称是华为集合通信库（Huawei Collective Communication Library），NCCL的全称是英伟达集合通信库（NVIDIA Collective Communication Library）。`init` 方法应该在 `set_context` 方法之后使用。

    **参数：**

    - **backend_name** (str) – 分布式后端的名称，可选HCCL或NCCL。如果未设置则根据硬件平台类型（device_target）进行推断，默认值为None。

    **异常：**

    - **TypeError** – 参数 `backend_name` 不是字符串。
    - **RuntimeError** – 1）硬件设备类型无效；2）后台服务无效；3）分布式计算初始化失败；4）未设置环境变量 `RANK_ID` 或 `MINDSPORE_HCCL_CONFIG_PATH` 的情况下初始化HCCL服务。
    - **ValueError** – 环境变量 `RANK_ID` 设置成非数字。

    **样例：**

    >>> from mindspore.context import set_context
    >>> set_context(device_target="Ascend")
    >>> init()

.. py:class:: mindspore.communication.release()

    释放分布式资源，例如‘HCCL’或‘NCCL’服务。

    .. note:: `release` 方法应该在 `init` 方法之后使用。

    **异常：**

    - **RuntimeError** - 在释放分布式资源失败时抛出。

.. py:class:: mindspore.communication.get_rank(group=GlobalComm.WORLD_COMM_GROUP)

    在指定通信组中获取当前的设备序号。

    .. note::`get_rank` 方法应该在 `init` 方法之后使用。

    **参数：**

    - **group** (str) - 通信组名称，通常由 `create_group` 方法创建，否则将使用默认组。默认值：‘GlobalComm.WORLD_COMM_GROUP’。

    **返回：**

    int, 调用该方法的进程对应的组内序号。

    **异常：**

    - **TypeError** – 在参数 `group` 不是字符串时抛出。
    - **ValueError** – 在后台不可用时抛出。
    - **RuntimeError** – 在‘HCCL’或‘NCCL’服务不可用时抛出。

.. py:class:: mindspore.communication.get_group_size(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组实例的rank_size。

    .. note:: `get_group_size` 方法应该在 `init` 方法之后使用。在跑用例之前用户需要预先配置通信相关的环境变量。

    **参数：**

    - **group** (str) - 指定工作组实例（由 create_group 方法创建）的名称，支持数据类型为str，默认值为‘WORLD_COMM_GROUP’。

    **返回：**

    指定通信组实例的rank_size，数据类型为int。

    **异常：**

    - **TypeError** – 在参数 `group` 不是字符串时抛出。
    - **ValueError** – 在后台不可用时抛出。
    - **RuntimeError** – 在‘HCCL’或‘NCCL’服务不可用时抛出。

    **样例：**

    >>> from mindspore.context import set_context
    >>> set_context(device_target="Ascend")
    >>> init()
    >>> group = "0-4"
    >>> rank_ids = [0,4]
    >>> create_group(group, rank_ids)
    >>> group_size = get_group_size(group)
    >>> print("group_size is: ", group_size)
    >>> group_size is:2

.. py:class:: mindspore.communication.get_world_rank_from_group_rank(group, group_rank_id)

    由指定通信组中的设备序号获取通信集群中的全局设备序号。

    .. note::
        - GPU 版本的MindSpore不支持此方法；
        - 参数 `group` 不能是 `hccl_world_group`；
        - `get_world_rank_from_group_rank` 方法应该在 `init` 方法之后使用。

    **参数：**

    - **group** (str) - 传入的通信组名称，通常由 `create_group` 方法创建。
    - **group_rank_id** (int) - 通信组内的设备序号。

    **返回：**

    int, 通信集群中的全局设备序号。

    **异常：**

    - **TypeError** – 参数 `group` 不是字符串或参数 `group_rank_id` 不是数字。
    - **ValueError** – 参数 `group` 是 `hccl_world_group` 或后台不可用。
    - **RuntimeError** – ‘HCCL’或‘NCCL’服务不可用，以及使用CPU版本的MindSpore。

    **样例：**

    >>> from mindspore.context import set_context
    >>> set_context(device_target="Ascend")
    >>> init()
    >>> group = "0-4"
    >>> rank_ids = [0,4]
    >>> create_group(group, rank_ids)
    >>> world_rank_id = get_world_rank_from_group_rank(group, 1)
    >>> print("world_rank_id is: ", world_rank_id) # 全局设备序号为4

.. py:class:: mindspore.communication.get_group_rank_from_world_rank(world_rank_id, group)

    由通信集群中的全局设备序号获取指定用户通信组中的rank ID。

    .. note::
        - GPU 版本的MindSpore不支持此方法；
        - 参数 `group` 不能是 `hccl_world_group`；
        - `get_group_rank_from_world_rank` 方法应该在 `init` 方法之后使用。

    **参数：**

    - **world_rank_id** (`int`) - 通信集群内的全局rank ID。
    - **group** (`str`) - 指定通信组实例（由 create_group 方法创建）的名称。

    **返回：**

    当前通信组内的rank_ID，数据类型为int。

    **异常：**

    - **TypeError** – 在参数 `group_rank_id` 不是数字或参数 `group` 不是字符串时抛出。
    - **ValueError** – 在参数 `group` 是 `hccl_world_group` 或后台不可用时抛出。
    - **RuntimeError** – 在‘HCCL’或‘NCCL’服务不可用，以及使用GPU版本的MindSpore时抛出。

    **样例：**

    >>> from mindspore.context import set_context
    >>> set_context(device_target="Ascend")
    >>> init()
    >>> group = "0-4"
    >>> rank_ids = [0,4]
    >>> create_group(group, rank_ids)
    >>> group_rank_id = get_group_rank_from_world_rank(4, group)
    >>> print("group_rank_id is: ", group_rank_id) # 组内设备序号是1

.. py:class:: mindspore.communication.create_group(group, rank_ids)

    创建用户自定义的通信组实例。

    .. note::
        - GPU 版本的MindSpore不支持此方法；
        - 列表rank_ids的长度应大于1；
        - 列表rank_ids内不能有重复数据；
        - `create_group` 方法应该在 `init` 方法之后使用。

    **参数：**

    - **group** (str) - 输入用户自定义的通信组实例名称，支持数据类型为str。
    - **rank_ids** (list) - 设备编号列表。

    **异常：**

    - **TypeError** – 参数 `group_rank_id` 不是数字或参数 `group` 不是字符串。
    - **ValueError** – 列表rank_ids的长度小于1，或列表rank_ids内有重复数据，以及后台无效。
    - **RuntimeError** – 在‘HCCL’或‘NCCL’ 服务不可用，以及使用CPU版本的MindSpore。

    **样例：**

    >>> from mindspore.context import set_context
    >>> set_context(device_target="Ascend")
    >>> init()
    >>> group = "0-8"
    >>> rank_ids = [0,8]
    >>> create_group(group, rank_ids)

.. py:class:: mindspore.communication.get_local_rank(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组中当前设备的本地设备序号。

    .. note::
        - GPU 版本的MindSpore不支持此方法；
        - `get_local_rank` 方法应该在 `init` 方法之后使用。

    **参数：**

    - **group** (`str`) - 通信组名称，通常由 `create_group` 方法创建，否则将使用默认组名称。默认值：‘WORLD_COMM_GROUP’。

    **返回：**

    int, 调用该方法的进程对应的通信组内本地设备序号。

    **异常：**

    - **TypeError** – 在参数 `group` 不是字符串时抛出。
    - **ValueError** – 在后台不可用时抛出。
    - **RuntimeError** – 在‘HCCL’或‘NCCL’服务不可用时抛出。

.. py:class:: mindspore.communication.get_local_rank_size(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组的本地设备总数。

    .. note::
        - GPU 版本的MindSpore不支持此方法；
        - `get_local_rank_size` 方法应该在 `init` 方法之后使用。

    **参数：**

    - **group** (str) - 传入的通信组名称，通常由 `create_group` 方法创建，或默认使用‘WORLD_COMM_GROUP’。

    **返回：**

    int, 调用该方法的进程对应的通信组设备总数。

    **异常：**

    - **TypeError** – 在参数 `group` 不是字符串时抛出。
    - **ValueError** – 在后台不可用时抛出。
    - **RuntimeError** – 在‘HCCL’或‘NCCL’服务不可用时抛出。

.. py:class:: mindspore.communication.destroy_group(group)

    注销用户通信组。

    .. note::
        - GPU 版本的MindSpore不支持此方法；
        - 参数 `group` 不能是 `hccl_world_group`；
        - `destroy_group` 方法应该在 `init` 方法之后使用。

    **参数：**

    - **group** (str) - 被注销通信组实例（通常由 create_group 方法创建）的名称。

    **异常：**

    - **TypeError** – 在参数 `group` 不是字符串时抛出。
    - **ValueError** – 在参数 `group` 是 `hccl_world_group` 或后台不可用时抛出。
    - **RuntimeError** – 在‘HCCL’或‘NCCL’服务不可用时抛出。

    **样例：**

    >>> from mindspore.context import set_context
    >>> set_context(device_target="Ascend")
    >>> init()
    >>> group = "0-8"
    >>> rank_ids = [0,8]
    >>> create_group(group, rank_ids)
    >>> destroy_group(group)
