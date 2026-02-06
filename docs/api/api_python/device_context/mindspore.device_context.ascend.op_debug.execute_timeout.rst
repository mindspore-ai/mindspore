mindspore.device_context.ascend.op_debug.execute_timeout
========================================================

.. py:function:: mindspore.device_context.ascend.op_debug.execute_timeout(op_timeout)

    设置一个算子的最大执行时间，以秒为单位。框架默认算子执行超时时间为900秒。
    详细信息请查看 `昇腾社区关于aclrtSetOpExecuteTimeOut文档说明 <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/appdevgapi/aclcppdevg_03_0132.html>`_。

    参数：
        - **op_timeout** (int) - 算子的最大执行时间。如果执行时间超过这个值，系统将终止该任务。0意味着使用默认值。AI Core和AI CPU算子在不同硬件上的默认值有差异。