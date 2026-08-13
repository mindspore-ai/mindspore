mindspore.device_context.ascend.op_precision.cube_math_type
===========================================================

.. py:function:: mindspore.device_context.ascend.op_precision.cube_math_type(value)

    配置Addmm和Baddbmm算子的Cube计算模式。该值会直接传递给ACLNN的 ``cubeMathType`` 参数。显式配置优先于现有自动精度策略；未配置时继续使用自动精度策略。

    ``4`` 表示FP16或BF16矩阵乘后使用FP32执行加法。该模式的支持范围和行为取决于输入shape、数据类型、硬件和CANN版本。对于Addmm和Baddbmm，MindSpore会适配相对矩阵乘结果需要广播的input张量，以满足底层ACLNN的shape要求。

    .. note::
        这是一个实验特性，可能会被更改或者删除。

    参数：
        - **value** (int) - Cube计算模式。支持 ``0`` （KEEP_DTYPE）、 ``1`` （ALLOW_FP32_DOWN_PRECISION）、
          ``2`` （FORCE_FP16）、 ``3`` （FORCE_HF32）和 ``4`` （USE_FP32_ADD）。
