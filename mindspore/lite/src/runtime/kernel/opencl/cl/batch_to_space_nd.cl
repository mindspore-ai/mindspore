#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void batch_to_space_nd_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 src_size,
                                      int4 dst_size, int2 block_size, int4 paddings) {
  int X = get_global_id(0);  // c
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // h*n
  if (X >= src_size.x || Y >= src_size.y || Y >= src_size.z) {
    return;
  }
  for (int i = 0; i < block_size.x; ++i) {
    for (int j = 0; j < block_size.y; ++j) {
      int Y_dst = (Y * block_size.y + j);
      int Z_dst = Z * block_size.x + i;
      if (Y_dst >= dst_size.y || Z_dst >= dst_size.z) {
        continue;
      }
      int Y_org = (Y_dst + paddings.z) / block_size.y;
      int Z_org = (Z_dst + paddings.x) / block_size.x;
      int Z_com = (i * block_size.y + j) * src_size.z + Z_org;
      FLT4 res_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
      res_data = READ_IMAGE(src_data, smp_zero, (int2)(Y_org * dst_size.x + X, Z_com));
      WRITE_IMAGE(dst_data, (int2)((Y * block_size.y + j) * dst_size.x + X, Z * block_size.x + i), res_data);
    }
  }
}
__kernel void batch_to_space_nd_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 src_size,
                                       int4 dst_size, int2 block_size, int4 paddings) {
  int X = get_global_id(0);  // c
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // h*n
  if (X >= dst_size.x || Y >= dst_size.y || Y >= dst_size.z) {
    return;
  }
  for (int i = 0; i < block_size.x; ++i) {
    for (int j = 0; j < block_size.y; ++j) {
      int Y_dst = (Y * block_size.y + j);
      int Z_dst = Z * block_size.x + i;
      int Y_org = (Y_dst + paddings.z) / block_size.y;
      int Z_org = (Z_dst + paddings.x) / block_size.x;
      int Z_com = (i * block_size.y + j) * src_size.z + Z_org;
      FLT4 res_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
      res_data = READ_IMAGE(src_data, smp_zero, (int2)(Y_org * dst_size.x + X, Z_com));
      WRITE_IMAGE(dst_data, (int2)((Y * block_size.y + j) * dst_size.x + X, Z * block_size.x + i), res_data);
    }
  }
}
