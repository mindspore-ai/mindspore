"""ResizeBicubicGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

resize_bicubic_grad_op_info = AiCPURegOp("ResizeBicubicGrad") \
    .fusion_type("OPAQUE") \
    .input(0, "grads", "required") \
    .input(1, "original_image", "required") \
    .output(0, "y", "required") \
    .attr("align_corners", "bool") \
    .attr("half_pixel_centers", "bool") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(resize_bicubic_grad_op_info)
def _resize_bicubic_grad_aicpu():
    """ResizeBicubicGrad AiCPU register"""
    return
