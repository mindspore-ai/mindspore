#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void to_format_NHWC_to_NHWC4_IMG(__global FLT4 *src_data, __write_only image2d_t dst_data, int4 size,
                                          int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global FLT *src_addr = (__global FLT *)src_data;
  src_addr += offset;
  FLT4 data = (FLT4)(0.f);
  if ((Z + 1) * 4 <= shape.w) {
    data = ((__global FLT4 *)src_addr)[0];
  } else {
    if ((shape.w - Z * 4) >= 1) {
      data.x = src_addr[0];
    }
    if ((shape.w - Z * 4) >= 2) {
      data.y = src_addr[1];
    }
    if ((shape.w - Z * 4) >= 3) {
      data.z = src_addr[2];
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), data);
}
__kernel void to_format_NHWC4_to_NHWC4_IMG(__global FLT4 *src_data, __write_only image2d_t dst_data, int4 size,
                                           int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), src_data[(X * size.y + Y) * size.z + Z]);
}
__kernel void to_format_NC4HW4_to_NC4HW4_IMG(__global FLT4 *src_data, __write_only image2d_t dst_data, int4 size,
                                             int4 shape) {
  // size(h, w, c4, 1), shape(n, c, h, w)
  int X = get_global_id(0);  // h
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c4
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  WRITE_IMAGE(dst_data, (int2)(Y, Z * size.x + X), src_data[(Z * size.x + X) * size.y + Y]);
}
__kernel void to_format_NCHW_to_NCHW_BUF(__read_only image2d_t src_data, __global FLT4 *dst_data, int4 size,
                                         int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(Z * size.y + Y) * size.x + X] = READ_IMAGE(src_data, smp_zero, (int2)(Y * size.x + X, Z));
}
__kernel void to_format_NHWC4_to_NHWC_BUF(__read_only image2d_t src_data, __global FLT4 *dst_data, int4 size,
                                          int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  FLT4 data = READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + Z, X));
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global FLT *dst_addr = (__global FLT *)dst_data;
  dst_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    ((__global FLT4 *)dst_addr)[0] = data;
  } else {
    if (shape.w - Z * 4 >= 1) {
      dst_addr[0] = data.x;
    }
    if (shape.w - Z * 4 >= 2) {
      dst_addr[1] = data.y;
    }
    if (shape.w - Z * 4 >= 3) {
      dst_addr[2] = data.z;
    }
  }
}
__kernel void to_format_NC4HW4_to_NC4HW4_BUF(__read_only image2d_t src_data, __global FLT4 *dst_data, int4 size,
                                             int4 shape) {
  // size(h, w, c, 1), shape(n, c, h, w)
  int X = get_global_id(0);  // h
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(Z * size.x + X) * size.y + Y] = READ_IMAGE(src_data, smp_zero, (int2)(Y, Z * size.x + X));
}
__kernel void to_format_NHWC4_to_NHWC4_BUF(__read_only image2d_t src_data, __global FLT4 *dst_data, int4 size,
                                           int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(X * size.y + Y) * size.z + Z] = READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + Z, X));
}
