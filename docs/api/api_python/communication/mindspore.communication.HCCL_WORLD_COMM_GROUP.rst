mindspore.communication.HCCL_WORLD_COMM_GROUP
=============================================

.. py:data:: mindspore.communication.HCCL_WORLD_COMM_GROUP

    ``"hccl_world_group"`` 字符串，指的是由HCCL创建的默认通信组。在Ascend硬件平台下，初始化通信服务后，该字符串与 ``GlobalComm.WORLD_COMM_GROUP`` 等价。推荐使用 ``GlobalComm.WORLD_COMM_GROUP`` 获取当前全局通信组。
