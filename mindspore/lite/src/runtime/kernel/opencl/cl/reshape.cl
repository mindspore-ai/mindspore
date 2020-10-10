#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void reshape_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, int4 size_out) {
  int X = get_global_id(0);
  if (X >= size_out.x * size_out.y * size_out.z * size_out.w) {
    return;
  }
  int in_img_x = size.z * size.w;
  int out_img_x = size_out.z * size_out.w;
  WRITE_IMAGE(dst_data, (int2)(X % out_img_x, X / out_img_x),
              READ_IMAGE(src_data, smp_zero, (int2)(X % in_img_x, X / in_img_x)));
}

__kernel void reshape_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size,
                             int4 size_out) {
  int X = get_global_id(0);
  if (X >= size_out.x * size_out.y * size_out.z * size_out.w) {
    return;
  }
  int in_img_x = size.z;
  int out_img_x = size_out.z;
  WRITE_IMAGE(dst_data, (int2)(X % out_img_x, X / out_img_x),
              READ_IMAGE(src_data, smp_zero, (int2)(X % in_img_x, X / in_img_x)));
}
