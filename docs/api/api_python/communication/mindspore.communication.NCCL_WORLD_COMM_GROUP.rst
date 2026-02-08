mindspore.communication.NCCL_WORLD_COMM_GROUP
=============================================

.. py:data:: mindspore.communication.NCCL_WORLD_COMM_GROUP

    ``"nccl_world_group"`` 字符串，指的是由NCCL创建的默认通信组。在GPU硬件平台下，初始化通信服务后，该字符串与 ``GlobalComm.WORLD_COMM_GROUP`` 等价。推荐使用 ``GlobalComm.WORLD_COMM_GROUP`` 获取当前全局通信组。
