#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void reshape(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, int4 size_out) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size_out.x || Y >= size_out.y || Z >= size_out.z) {
    return;
  }
  int out_index = X * size_out.y + Y;
  int ih = out_index / size.y;
  int iw = out_index % size.y;
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), READ_IMAGE(src_data, smp_zero, (int2)(iw * size.z + Z, ih)));
}
