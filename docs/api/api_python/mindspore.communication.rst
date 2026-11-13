mindspore.communication
========================

集合通信接口。

注意，集合通信接口需要先配置好通信环境变量。

针对Ascend/GPU/CPU设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html>`_ 。

.. mscnautosummary::
    :toctree: communication
    :nosignatures:
    :template: classtemplate.rst

    mindspore.communication.GlobalComm
    mindspore.communication.init
    mindspore.communication.release
    mindspore.communication.create_group
    mindspore.communication.destroy_group
    mindspore.communication.get_comm_name
    mindspore.communication.get_group_size
    mindspore.communication.get_group_rank_from_world_rank
    mindspore.communication.get_local_rank
    mindspore.communication.get_local_rank_size
    mindspore.communication.get_process_group_ranks
    mindspore.communication.get_rank
    mindspore.communication.get_world_rank_from_group_rank
    mindspore.communication.HCCL_WORLD_COMM_GROUP
    mindspore.communication.NCCL_WORLD_COMM_GROUP
    mindspore.communication.MCCL_WORLD_COMM_GROUP

mindspore.communication.comm_func
---------------------------------

集合通信函数式接口。

.. mscnautosummary::
    :toctree: communication
    :nosignatures:
    :template: classtemplate.rst

    mindspore.communication.comm_func.all_gather_into_tensor
    mindspore.communication.comm_func.all_reduce
    mindspore.communication.comm_func.barrier
    mindspore.communication.comm_func.broadcast
    mindspore.communication.comm_func.gather_into_tensor
    mindspore.communication.comm_func.reduce
    mindspore.communication.comm_func.scatter_tensor
