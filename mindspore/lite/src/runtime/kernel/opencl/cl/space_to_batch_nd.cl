#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void space_to_batch_nd_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 src_size,
                                      int4 dst_size, int2 block_size, int4 paddings) {
  int X = get_global_id(0);  // c
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // h * n_i
  // (N,H*BH,W*BW,C) to (BH*BW*N,H,W,C)
  int N_I = Z / dst_size.z;
  Z = Z % dst_size.z;
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z || N_I >= src_size.w) {
    return;
  }
  for (int i = 0; i < block_size.x; ++i) {
    for (int j = 0; j < block_size.y; ++j) {
      int w_org = Y * block_size.y + j - paddings.z;
      int h_org = Z * block_size.x + i - paddings.x;
      FLT4 res_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
      if (h_org >= 0 && h_org < src_size.z)
        res_data = READ_IMAGE(src_data, smp_zero, (int2)(w_org * dst_size.x + X, N_I * src_size.z + h_org));
      WRITE_IMAGE(dst_data, (int2)(Y * dst_size.x + X, ((i * block_size.y + j) * src_size.w + N_I) * dst_size.z + Z),
                  res_data);
    }
  }
}
__kernel void space_to_batch_nd_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 src_size,
                                       int4 dst_size, int2 block_size, int4 paddings) {
  int X = get_global_id(0);  // c
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // h
  if (X >= dst_size.x || Y >= dst_size.y || Y >= dst_size.z) {
    return;
  }
  for (int i = 0; i < block_size.x; ++i) {
    for (int j = 0; j < block_size.y; ++j) {
      int w_org = Y * block_size.y + j - paddings.z;
      int h_org = Z * block_size.x + i - paddings.x;
      FLT4 res_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
      if (w_org >= 0 && w_org < src_size.y && h_org >= 0 && h_org < src_size.z) {
        res_data = READ_IMAGE(src_data, smp_zero, (int2)(h_org * src_size.y + Y, X));
      }
      WRITE_IMAGE(dst_data, (int2)(Z * dst_size.y + Y, (i * block_size.y + j) * dst_size.x + X), res_data);
    }
  }
}
