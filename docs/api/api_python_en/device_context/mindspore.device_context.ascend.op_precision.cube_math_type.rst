mindspore.device_context.ascend.op_precision.cube_math_type
===========================================================

.. py:function:: mindspore.device_context.ascend.op_precision.cube_math_type(value)

    Configures the Cube calculation mode for Addmm and Baddbmm operators. The value is passed directly to the ACLNN
    ``cubeMathType`` parameter. An explicitly configured value takes precedence over the existing automatic precision
    policy. If it is not configured, the automatic precision policy is retained.

    Mode ``4`` performs the addition in FP32 after FP16 or BF16 matrix multiplication. Its support and behavior depend
    on the input shape, data type, hardware, and CANN version. When Addmm or Baddbmm uses this mode, the input tensor
    cannot be broadcast relative to the matrix multiplication result.

    .. note::
        This is an experimental prototype that is subject to change and/or deletion.

    Parameters:
        - **value** (int) - The Cube calculation mode. Supported values are ``0`` (KEEP_DTYPE),
          ``1`` (ALLOW_FP32_DOWN_PRECISION), ``2`` (FORCE_FP16), ``3`` (FORCE_HF32), and ``4`` (USE_FP32_ADD).
