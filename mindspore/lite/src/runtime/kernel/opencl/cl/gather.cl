#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void gather_NHWC4(__read_only image2d_t src_data, __global int *indices, __write_only image2d_t dst_data,
                           int4 src_size, int4 dst_size, int indices_num, int axis) {
  int X = get_global_id(0);  // w
  int Y = get_global_id(1);  // h
  int Z = get_global_id(2);  // c
  if (X >= dst_size.x || Y >= dst_size.y || Y >= dst_size.z) {
    return;
  }
  FLT4 res_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  if (axis == 1) {
    res_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + Z, indices[Y]));
  } else if (axis == 2) {
    res_data = READ_IMAGE(src_data, smp_zero, (int2)(indices[X] * src_size.z + Z, Y));
  } else if (axis == 3) {
    int offset[4] = {indices[Z * 4] / 4, indices[Z * 4 + 1] / 4, indices[Z * 4 + 2] / 4, indices[Z * 4 + 3] / 4};
    FLT tmp[4];
    FLT res_tmp[4];
    for (int i = 0; i < 4; ++i) {
      if (i >= 1 && offset[i] != offset[i - 1]) {
        FLT4 rd_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
        rd_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + offset[i], Y));
        tmp[0] = rd_data.x;
        tmp[1] = rd_data.y;
        tmp[2] = rd_data.z;
        tmp[3] = rd_data.w;
      }
      res_tmp[i] = tmp[indices[Z * 4 + i] % 4];
    }
    res_data.x = res_tmp[0];
    res_data.y = res_tmp[1];
    res_data.z = res_tmp[2];
    res_data.w = res_tmp[3];
  }
  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), res_data);
}
__kernel void gather_NC4HW4(__read_only image2d_t src_data, __global int *indices, __write_only image2d_t dst_data,
                           int4 src_size, int4 dst_size, int indices_num, int axis) {
  int X = get_global_id(0);  // w
  int Y = get_global_id(1);  // h
  int Z = get_global_id(2);  // c
  if (X >= dst_size.x || Y >= dst_size.y || Y >= dst_size.z) {
    return;
  }
  FLT4 res_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  if (axis == 1) {
    res_data = READ_IMAGE(src_data, smp_zero, (int2)(X, Z * dst_size.y + indices[Y]));
  } else if (axis == 2) {
    res_data = READ_IMAGE(src_data, smp_zero, (int2)(indices[X], Z * dst_size.y + Y));
  } else if (axis == 3) {
    int offset[4] = {indices[Z * 4] / 4, indices[Z * 4 + 1] / 4, indices[Z * 4 + 2] / 4, indices[Z * 4 + 3] / 4};
    FLT tmp[4];
    FLT res_tmp[4];
    for (int i = 0; i < 4; ++i) {
      if (i >= 1 && offset[i] != offset[i - 1]) {
        FLT4 rd_data = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
        rd_data = READ_IMAGE(src_data, smp_zero, (int2)(X, offset[i] * dst_size.y + Y));
        tmp[0] = rd_data.x;
        tmp[1] = rd_data.y;
        tmp[2] = rd_data.z;
        tmp[3] = rd_data.w;
      }
      res_tmp[i] = tmp[indices[Z * 4 + i] % 4];
    }
    res_data.x = res_tmp[0];
    res_data.y = res_tmp[1];
    res_data.z = res_tmp[2];
    res_data.w = res_tmp[3];
  }
  WRITE_IMAGE(dst_data, (int2)(X, (Z * dst_size.y + Y)), res_data);
}
