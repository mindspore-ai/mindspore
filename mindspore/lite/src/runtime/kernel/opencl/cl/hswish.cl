#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void hswish(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 tensor_shape) {
  int X = get_global_id(0);  // n*h n: default =1
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c
  if (X >= tensor_shape.x * tensor_shape.y || Y >= tensor_shape.z || Z >= tensor_shape.w || tensor_shape.y == 0) {
    return;
  }
  int n = X / tensor_shape.y;
  int h = X % tensor_shape.y;
  FLT4 temp = READ_IMAGE(src_data, smp_none, (int2)((Y)*tensor_shape.w + Z, (n * tensor_shape.y + h)));
  FLT4 result = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  result.x = temp.x <= -3 ? 0 : (temp.x >= 3 ? 1 : temp.x / 6 + 0.5f);
  result.y = temp.y <= -3 ? 0 : (temp.y >= 3 ? 1 : temp.y / 6 + 0.5f);
  result.z = temp.z <= -3 ? 0 : (temp.z >= 3 ? 1 : temp.z / 6 + 0.5f);
  result.w = temp.w <= -3 ? 0 : (temp.w >= 3 ? 1 : temp.w / 6 + 0.5f);
  WRITE_IMAGE(dst_data, (int2)((Y)*tensor_shape.w + Z, (n * tensor_shape.y + h)), result);
}
