mindspore.communication.GlobalComm
==================================

.. py:class:: mindspore.communication.GlobalComm

    GlobalComm 是一个储存通信信息的全局类。成员包含： ``BACKEND`` 、 ``WORLD_COMM_GROUP`` 、 ``INITED`` 、 ``CHECK_ENVS`` 。

    - ``BACKEND`` ：使用的通信库， ``"hccl"`` 、 ``"nccl"`` 或者 ``"mccl"`` 。 ``"hccl"`` 代表华为集合通信库HCCL， ``"nccl"`` 代表英伟达集合通信库NCCL， ``"mccl"`` 代表MindSpore集合通信库MCCL。
    - ``WORLD_COMM_GROUP`` ：全局通信域， ``"hccl_world_group"`` 、 ``"nccl_world_group"`` 或者 ``"mccl_world_group"`` 。
    - ``INITED`` ：分布式通信是否已初始化。
    - ``CHECK_ENVS`` ：是否检查环境变量。