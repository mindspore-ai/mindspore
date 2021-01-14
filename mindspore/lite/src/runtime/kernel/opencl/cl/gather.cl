#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gather(__write_only image2d_t dst_data, __read_only image2d_t src_data, __global int *indices,
                     int4 src_size, int4 dst_size, int indices_num, int axis) {
  int X = get_global_id(0);  // w
  int Y = get_global_id(1);  // n*h
  int Z = get_global_id(2);  // c
  if (X >= dst_size.x || Y >= dst_size.y * dst_size.w || Z >= dst_size.z || dst_size.y == 0) {
    return;
  }
  FLT4 res_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int batch = Y / dst_size.y;
  int height = Y % dst_size.y;
  if (axis == 0) {
    res_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + Z, indices[batch] * src_size.y + height));
  } else if (axis == 1) {
    res_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + Z, batch * src_size.y + indices[height]));
  } else if (axis == 2) {
    res_data = READ_IMAGE(src_data, smp_zero, (int2)(indices[X] * src_size.z + Z, batch * src_size.y + height));
  } else if (axis == 3) {
    int offset[4] = {indices[Z * 4] / 4, indices[Z * 4 + 1] / 4, indices[Z * 4 + 2] / 4, indices[Z * 4 + 3] / 4};
    FLT tmp[4];
    FLT res_tmp[4];
    for (int i = 0; i < indices_num; ++i) {
      FLT4 rd_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
      rd_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + offset[i], batch * src_size.y + height));
      if (i >= 1 && offset[i] != offset[i - 1]) {
        rd_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + offset[i], batch * src_size.y + height));
      }
      tmp[0] = rd_data.x;
      tmp[1] = rd_data.y;
      tmp[2] = rd_data.z;
      tmp[3] = rd_data.w;
      res_tmp[i] = tmp[indices[Z * 4 + i] % 4];
    }
    res_data.x = res_tmp[0];
    res_data.y = res_tmp[1];
    res_data.z = res_tmp[2];
    res_data.w = res_tmp[3];
  }
  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, batch * dst_size.y + height), res_data);
}
