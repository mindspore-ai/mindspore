mindspore.device_context
===========================

device_context中封装了设备数量查询与当前指定后端是否可用的接口。

CPU设备后端管理
-------------------------

.. mscnautosummary::
    :toctree: device_context
    :nosignatures:
    :template: classtemplate.rst

    mindspore.device_context.cpu.device_count
    mindspore.device_context.cpu.is_available
    mindspore.device_context.cpu.op_tuning.threads_num


GPU设备后端管理
-------------------------

.. mscnautosummary::
    :toctree: device_context
    :nosignatures:
    :template: classtemplate.rst

    mindspore.device_context.gpu.device_count
    mindspore.device_context.gpu.is_available
    mindspore.device_context.gpu.op_precision.conv_allow_tf32
    mindspore.device_context.gpu.op_precision.matmul_allow_tf32
    mindspore.device_context.gpu.op_tuning.conv_dgrad_algo
    mindspore.device_context.gpu.op_tuning.conv_fprop_algo
    mindspore.device_context.gpu.op_tuning.conv_wgrad_algo

Ascend设备后端管理
-------------------------

.. mscnautosummary::
    :toctree: device_context
    :nosignatures:
    :template: classtemplate.rst

    mindspore.device_context.ascend.device_count
    mindspore.device_context.ascend.is_available
    mindspore.device_context.ascend.op_precision.conv_allow_hf32
    mindspore.device_context.ascend.op_precision.cube_math_type
    mindspore.device_context.ascend.op_precision.matmul_allow_hf32
    mindspore.device_context.ascend.op_precision.precision_mode
    mindspore.device_context.ascend.op_precision.op_precision_mode
    mindspore.device_context.ascend.op_debug.execute_timeout
    mindspore.device_context.ascend.op_debug.debug_option
    mindspore.device_context.ascend.op_debug.aclinit_config
    mindspore.device_context.ascend.op_tuning.op_compile
    mindspore.device_context.ascend.op_tuning.aclnn_cache
    mindspore.device_context.ascend.op_tuning.aoe_tune_mode
    mindspore.device_context.ascend.op_tuning.aoe_job_type
