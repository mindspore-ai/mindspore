#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void to_format_NHWC_to_NHWC4_IMG_float(__global float4 *src_data, __write_only image2d_t dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  FLT4 data = (FLT4)(0.f);
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global float *src_addr = (__global float *)src_data;
  src_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    data = TO_FLT4(((__global float4 *)src_addr)[0]);
  } else {
    if ((shape.w - Z * 4) >= 1) {
      data.x = (FLT)src_addr[0];
    }
    if ((shape.w - Z * 4) >= 2) {
      data.y = (FLT)src_addr[1];
    }
    if ((shape.w - Z * 4) >= 3) {
      data.z = (FLT)src_addr[2];
    }
  }
  if (size.y * size.z <= MAX_IMAGE2D_WIDTH)
    WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), data);
  else
    WRITE_IMAGE(dst_data, (int2)(Z, X * size.y + Y), data);
}
__kernel void to_format_NHWC_to_NHWC4_IMG_half(__global half4 *src_data, __write_only image2d_t dst_data, int4 size,
                                               int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  FLT4 data = (FLT4)(0.f);
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global half *src_addr = (__global half *)src_data;
  src_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    data = TO_FLT4(((__global half4 *)src_addr)[0]);
  } else {
    if ((shape.w - Z * 4) >= 1) {
      data.x = (FLT)src_addr[0];
    }
    if ((shape.w - Z * 4) >= 2) {
      data.y = (FLT)src_addr[1];
    }
    if ((shape.w - Z * 4) >= 3) {
      data.z = (FLT)src_addr[2];
    }
  }
  if (size.y * size.z <= MAX_IMAGE2D_WIDTH)
    WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), data);
  else
    WRITE_IMAGE(dst_data, (int2)(Z, X * size.y + Y), data);
}
__kernel void to_format_NCHW_to_NHWC4_IMG_float(__global float4 *src_data, __write_only image2d_t dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  FLT4 data = (FLT4)(0.f);
  __global float *src_addr = (__global float *)src_data;
  __global float *src_addr_0 = src_addr + ((Z * 4 + 0) * shape.y + X) * shape.z + Y;
  __global float *src_addr_1 = src_addr + ((Z * 4 + 1) * shape.y + X) * shape.z + Y;
  __global float *src_addr_2 = src_addr + ((Z * 4 + 2) * shape.y + X) * shape.z + Y;
  if ((Z + 1) * 4 <= shape.w) {
    data = TO_FLT4(((__global float4 *)src_addr_0)[0]);
  } else {
    if ((shape.w - Z * 4) >= 1) {
      data.x = src_addr_0[0];
    }
    if ((shape.w - Z * 4) >= 2) {
      data.y = src_addr_1[0];
    }
    if ((shape.w - Z * 4) >= 3) {
      data.z = src_addr_2[0];
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), data);
}
__kernel void to_format_NCHW_to_NHWC4_IMG_half(__global half4 *src_data, __write_only image2d_t dst_data, int4 size,
                                               int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  FLT4 data = (FLT4)(0.f);
  __global half *src_addr = (__global half *)src_data;
  __global half *src_addr_0 = src_addr + ((Z * 4 + 0) * shape.y + X) * shape.z + Y;
  __global half *src_addr_1 = src_addr + ((Z * 4 + 1) * shape.y + X) * shape.z + Y;
  __global half *src_addr_2 = src_addr + ((Z * 4 + 2) * shape.y + X) * shape.z + Y;
  if ((Z + 1) * 4 <= shape.w) {
    data = TO_FLT4(((__global half4 *)src_addr_0)[0]);
  } else {
    if ((shape.w - Z * 4) >= 1) {
      data.x = src_addr_0[0];
    }
    if ((shape.w - Z * 4) >= 2) {
      data.y = src_addr_1[0];
    }
    if ((shape.w - Z * 4) >= 3) {
      data.z = src_addr_2[0];
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), data);
}
__kernel void to_format_NHWC_to_NC4HW4_IMG_float(__global float4 *src_data, __write_only image2d_t dst_data, int4 size,
                                                 int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z || shape.y == 0) {
    return;
  }
  int offset = (X / shape.y) * shape.y * shape.z * shape.w + ((X % shape.y) * shape.z + Y) * shape.w + Z * 4;
  __global float *src_addr = (__global float *)src_data;
  src_addr += offset;
  FLT4 data = (FLT4)(0.f);
  if ((Z + 1) * 4 <= shape.w) {
    data = TO_FLT4(((__global float4 *)src_addr)[0]);
  } else {
    if ((shape.w - Z * 4) >= 1) {
      data.x = (FLT)src_addr[0];
    }
    if ((shape.w - Z * 4) >= 2) {
      data.y = (FLT)src_addr[1];
    }
    if ((shape.w - Z * 4) >= 3) {
      data.z = (FLT)src_addr[2];
    }
  }
  int pos_ix = (X / shape.y) * size.z * shape.y + Z * shape.y + X % shape.y;
  WRITE_IMAGE(dst_data, (int2)(Y, pos_ix), data);
}
__kernel void to_format_NHWC_to_NC4HW4_IMG_half(__global half4 *src_data, __write_only image2d_t dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z || shape.y == 0) {
    return;
  }
  int offset = (X / shape.y) * shape.y * shape.z * shape.w + ((X % shape.y) * shape.z + Y) * shape.w + Z * 4;
  __global half *src_addr = (__global half *)src_data;
  src_addr += offset;
  FLT4 data = (FLT4)(0.f);
  if ((Z + 1) * 4 <= shape.w) {
    data = TO_FLT4(((__global half4 *)src_addr)[0]);
  } else {
    if ((shape.w - Z * 4) >= 1) {
      data.x = (FLT)src_addr[0];
    }
    if ((shape.w - Z * 4) >= 2) {
      data.y = (FLT)src_addr[1];
    }
    if ((shape.w - Z * 4) >= 3) {
      data.z = (FLT)src_addr[2];
    }
  }
  int pos_ix = (X / shape.y) * size.z * shape.y + Z * shape.y + X % shape.y;
  WRITE_IMAGE(dst_data, (int2)(Y, pos_ix), data);
}
__kernel void to_format_NHWC4_to_NHWC4_IMG_float(__global float4 *src_data, __write_only image2d_t dst_data, int4 size,
                                                 int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), TO_FLT4(src_data[(X * size.y + Y) * size.z + Z]));
}
__kernel void to_format_NHWC4_to_NHWC4_IMG_half(__global half4 *src_data, __write_only image2d_t dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), TO_FLT4(src_data[(X * size.y + Y) * size.z + Z]));
}
__kernel void to_format_NC4HW4_to_NC4HW4_IMG_float(__global float4 *src_data, __write_only image2d_t dst_data,
                                                   int4 size, int4 shape) {
  // size(h, w, c4, 1), shape(n, c, h, w)
  int X = get_global_id(0);  // h
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c4
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  WRITE_IMAGE(dst_data, (int2)(Y, Z * size.x + X), TO_FLT4(src_data[(Z * size.x + X) * size.y + Y]));
}
__kernel void to_format_NC4HW4_to_NC4HW4_IMG_half(__global half4 *src_data, __write_only image2d_t dst_data, int4 size,
                                                  int4 shape) {
  // size(h, w, c4, 1), shape(n, c, h, w)
  int X = get_global_id(0);  // h
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c4
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  WRITE_IMAGE(dst_data, (int2)(Y, Z * size.x + X), TO_FLT4(src_data[(Z * size.x + X) * size.y + Y]));
}
__kernel void to_format_NCHW_to_NCHW_BUF_float(__read_only image2d_t src_data, __global float4 *dst_data, int4 size,
                                               int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(Z * size.y + Y) * size.x + X] = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.x + X, Z)));
}
__kernel void to_format_NCHW_to_NCHW_BUF_half(__read_only image2d_t src_data, __global half4 *dst_data, int4 size,
                                              int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(Z * size.y + Y) * size.x + X] = convert_half4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.x + X, Z)));
}
__kernel void to_format_NHWC4_to_NHWC_BUF_float(__read_only image2d_t src_data, __global float4 *dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  float4 data;
  if (size.y * size.z <= MAX_IMAGE2D_WIDTH)
    data = convert_float4(READ_IMAGEIN(src_data, smp_zero, (int2)(Y * size.z + Z, X)));
  else
    data = convert_float4(READ_IMAGEIN(src_data, smp_zero, (int2)(Z, X * size.y + Y)));
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global float *dst_addr = (__global float *)dst_data;
  dst_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    ((__global float4 *)dst_addr)[0] = data;
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
__kernel void to_format_NHWC4_to_NCHW_BUF_float(__read_only image2d_t src_data, __global float4 *dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  float4 data = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + Z, X)));
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global float *dst_addr = (__global float *)dst_data;
  __global float *dst_addr_0 = dst_addr + ((Z * 4 + 0) * shape.y + X) * shape.z + Y;
  __global float *dst_addr_1 = dst_addr + ((Z * 4 + 1) * shape.y + X) * shape.z + Y;
  __global float *dst_addr_2 = dst_addr + ((Z * 4 + 2) * shape.y + X) * shape.z + Y;
  dst_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    ((__global float4 *)dst_addr_0)[0] = data;
  } else {
    if (shape.w - Z * 4 >= 1) {
      dst_addr_0[0] = data.x;
    }
    if (shape.w - Z * 4 >= 2) {
      dst_addr_1[0] = data.y;
    }
    if (shape.w - Z * 4 >= 3) {
      dst_addr_2[0] = data.z;
    }
  }
}
__kernel void to_format_NHWC4_to_NCHW_BUF_half(__read_only image2d_t src_data, __global half4 *dst_data, int4 size,
                                               int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  half4 data = convert_half4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + Z, X)));
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global half *dst_addr = (__global half *)dst_data;
  __global half *dst_addr_0 = dst_addr + ((Z * 4 + 0) * shape.y + X) * shape.z + Y;
  __global half *dst_addr_1 = dst_addr + ((Z * 4 + 1) * shape.y + X) * shape.z + Y;
  __global half *dst_addr_2 = dst_addr + ((Z * 4 + 2) * shape.y + X) * shape.z + Y;
  dst_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    ((__global half4 *)dst_addr_0)[0] = data;
  } else {
    if (shape.w - Z * 4 >= 1) {
      dst_addr_0[0] = data.x;
    }
    if (shape.w - Z * 4 >= 2) {
      dst_addr_1[0] = data.y;
    }
    if (shape.w - Z * 4 >= 3) {
      dst_addr_2[0] = data.z;
    }
  }
}
__kernel void to_format_NHWC4_to_NHWC_BUF_half(__read_only image2d_t src_data, __global half4 *dst_data, int4 size,
                                               int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  half4 data;
  if (size.y * size.z <= MAX_IMAGE2D_WIDTH)
    data = convert_half4(READ_IMAGEIN(src_data, smp_zero, (int2)(Y * size.z + Z, X)));
  else
    data = convert_half4(READ_IMAGEIN(src_data, smp_zero, (int2)(Z, X * size.y + Y)));
  int offset = (X * shape.z + Y) * shape.w + Z * 4;
  __global half *dst_addr = (__global half *)dst_data;
  dst_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    ((__global half4 *)dst_addr)[0] = data;
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
__kernel void to_format_NC4HW4_to_NHWC_BUF_float(__read_only image2d_t src_data, __global float4 *dst_data, int4 size,
                                                 int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z || shape.y == 0) {
    return;
  }
  int pos_ix = (X / shape.y) * size.z * shape.y + Z * shape.y + X % shape.y;
  float4 data = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(Y, pos_ix)));
  int offset = (X / shape.y) * shape.y * shape.z * shape.w + ((X % shape.y) * shape.z + Y) * shape.w + Z * 4;
  __global float *dst_addr = (__global float *)dst_data;
  dst_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    ((__global float4 *)dst_addr)[0] = data;
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
__kernel void to_format_NC4HW4_to_NHWC_BUF_half(__read_only image2d_t src_data, __global half4 *dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z || shape.y == 0) {
    return;
  }
  int pos_ix = (X / shape.y) * size.z * shape.y + Z * shape.y + X % shape.y;
  half4 data = convert_half4(READ_IMAGE(src_data, smp_zero, (int2)(Y, pos_ix)));
  int offset = (X / shape.y) * shape.y * shape.z * shape.w + ((X % shape.y) * shape.z + Y) * shape.w + Z * 4;
  __global half *dst_addr = (__global half *)dst_data;
  dst_addr += offset;
  if ((Z + 1) * 4 <= shape.w) {
    ((__global half4 *)dst_addr)[0] = data;
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
__kernel void to_format_NC4HW4_to_NC4HW4_BUF_float(__read_only image2d_t src_data, __global float4 *dst_data, int4 size,
                                                   int4 shape) {
  // size(h, w, c, 1), shape(n, c, h, w)
  int X = get_global_id(0);  // h
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(Z * size.x + X) * size.y + Y] = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(Y, Z * size.x + X)));
}
__kernel void to_format_NC4HW4_to_NC4HW4_BUF_half(__read_only image2d_t src_data, __global half4 *dst_data, int4 size,
                                                  int4 shape) {
  // size(h, w, c, 1), shape(n, c, h, w)
  int X = get_global_id(0);  // h
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(Z * size.x + X) * size.y + Y] = convert_half4(READ_IMAGE(src_data, smp_zero, (int2)(Y, Z * size.x + X)));
}
__kernel void to_format_NHWC4_to_NHWC4_BUF_float(__read_only image2d_t src_data, __global float4 *dst_data, int4 size,
                                                 int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(X * size.y + Y) * size.z + Z] = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + Z, X)));
}
__kernel void to_format_NHWC4_to_NHWC4_BUF_half(__read_only image2d_t src_data, __global half4 *dst_data, int4 size,
                                                int4 shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  dst_data[(X * size.y + Y) * size.z + Z] = convert_half4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + Z, X)));
}
