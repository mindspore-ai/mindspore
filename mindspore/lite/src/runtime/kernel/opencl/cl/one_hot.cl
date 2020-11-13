#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void OneHotAxis0(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int in_index = (H * out_shape.z + Y) * out_shape.w + X;
  FLT4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  FLT4 result = (FLT4)(0.f);
  if (4 * X < C) {
    if (indices_int[0] == N) {
      result.x = (FLT)(on_value);
    } else {
      result.x = (FLT)(off_value);
    }
  }
  if (4 * X + 1 < C) {
    if (indices_int[1] == N) {
      result.y = (FLT)(on_value);
    } else {
      result.y = (FLT)(off_value);
    }
  }
  if (4 * X + 2 < C) {
    if (indices_int[2] == N) {
      result.z = (FLT)(on_value);
    } else {
      result.z = (FLT)(off_value);
    }
  }
  if (4 * X + 3 < C) {
    if (indices_int[3] == N) {
      result.w = (FLT)(on_value);
    } else {
      result.w = (FLT)(off_value);
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHotAxis1(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int in_index = (N * out_shape.z + Y) * out_shape.w + X;
  FLT4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  FLT4 result = (FLT4)(0.f);
  if (4 * X < C) {
    if (indices_int[0] == H) {
      result.x = (FLT)(on_value);
    } else {
      result.x = (FLT)(off_value);
    }
  }
  if (4 * X + 1 < C) {
    if (indices_int[1] == H) {
      result.y = (FLT)(on_value);
    } else {
      result.y = (FLT)(off_value);
    }
  }
  if (4 * X + 2 < C) {
    if (indices_int[2] == H) {
      result.z = (FLT)(on_value);
    } else {
      result.z = (FLT)(off_value);
    }
  }
  if (4 * X + 3 < C) {
    if (indices_int[3] == H) {
      result.w = (FLT)(on_value);
    } else {
      result.w = (FLT)(off_value);
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHotAxis2(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int in_index = (N * out_shape.y + H) * out_shape.w + X;
  FLT4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  FLT4 result = (FLT4)(0.f);
  if (4 * X < C) {
    if (indices_int[0] == Y) {
      result.x = (FLT)(on_value);
    } else {
      result.x = (FLT)(off_value);
    }
  }
  if (4 * X + 1 < C) {
    if (indices_int[1] == Y) {
      result.y = (FLT)(on_value);
    } else {
      result.y = (FLT)(off_value);
    }
  }
  if (4 * X + 2 < C) {
    if (indices_int[2] == Y) {
      result.z = (FLT)(on_value);
    } else {
      result.z = (FLT)(off_value);
    }
  }
  if (4 * X + 3 < C) {
    if (indices_int[3] == Y) {
      result.w = (FLT)(on_value);
    } else {
      result.w = (FLT)(off_value);
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHotAxis3(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int ci4_size = UP_DIV(out_shape.z, C4NUM);
  int in_index_c4 = (N * out_shape.y + H) * ci4_size + Y / 4;
  int in_index_c4_remainder = Y % 4;
  FLT4 indices =
    READ_IMAGE(src_data, smp_zero, (int2)(in_index_c4 % in_image2d_shape.x, in_index_c4 / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  int index_one = indices_int[in_index_c4_remainder];
  FLT4 result = (FLT4)(0.f);
  if (4 * X < C) {
    if (index_one == 4 * X) {
      result.x = (FLT)(on_value);
    } else {
      result.x = (FLT)(off_value);
    }
  }
  if (4 * X + 1 < C) {
    if (index_one == 4 * X + 1) {
      result.y = (FLT)(on_value);
    } else {
      result.y = (FLT)(off_value);
    }
  }
  if (4 * X + 2 < C) {
    if (index_one == 4 * X + 2) {
      result.z = (FLT)(on_value);
    } else {
      result.z = (FLT)(off_value);
    }
  }
  if (4 * X + 3 < C) {
    if (index_one == 4 * X + 3) {
      result.w = (FLT)(on_value);
    } else {
      result.w = (FLT)(off_value);
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHot2DAxis0(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                            int4 out_shape, int depth, float on_value, float off_value, int C) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  FLT4 result = (FLT4)(0.f);
  int channel = 4 * X;
  if (channel < C) {
    FLT4 indices = READ_IMAGE(src_data, smp_zero, (int2)(0, channel));
    int index = ((int *)&indices)[0];
    if (index == Z) {
      result.x = (FLT)(on_value);
    } else {
      result.x = (FLT)(off_value);
    }
  }
  channel++;
  if (channel < C) {
    FLT4 indices = READ_IMAGE(src_data, smp_zero, (int2)(0, channel));
    int index = ((int *)&indices)[0];
    if (index == Z) {
      result.y = (FLT)(on_value);
    } else {
      result.y = (FLT)(off_value);
    }
  }
  channel++;
  if (channel < C) {
    FLT4 indices = READ_IMAGE(src_data, smp_zero, (int2)(0, channel));
    int index = ((int *)&indices)[0];
    if (index == Z) {
      result.z = (FLT)(on_value);
    } else {
      result.z = (FLT)(off_value);
    }
  }
  channel++;
  if (channel < C) {
    FLT4 indices = READ_IMAGE(src_data, smp_zero, (int2)(0, channel));
    int index = ((int *)&indices)[0];
    if (index == Z) {
      result.w = (FLT)(on_value);
    } else {
      result.w = (FLT)(off_value);
    }
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}
