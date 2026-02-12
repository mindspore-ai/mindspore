mindspore.device_context.ascend.op_tuning.aoe_job_type
======================================================

.. py:function:: mindspore.device_context.ascend.op_tuning.aoe_job_type(config)

    设置AOE工具专用的参数，需要和 :func:`mindspore.device_context.ascend.op_tuning.aoe_tune_mode` 配合使用。框架默认设置为 ``"2"``。

    参数：
        - **config** (str) - 选择调优类型。

          - ``"1"``: 设置为子图调优。
          - ``"2"``: 设置为算子调优。