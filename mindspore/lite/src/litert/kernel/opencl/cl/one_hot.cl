#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

#define SET_ON_OR_OFF_VALUE(RESULT, POSITION, INDICES, ON_VALUE, OFF_VALUE) \
  if (POSITION == INDICES) {                                                \
    RESULT = (float)(ON_VALUE);                                             \
  } else {                                                                  \
    RESULT = (float)(OFF_VALUE);                                            \
  }

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void OneHotAxis0(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int in_index = (H * out_shape.z + Y) * out_shape.w + X;
  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  for (int i = 0; i < C4NUM; i++) {
    if (support_neg_index != 0 && indices_int[i] < 0) {
      indices_int[i] += depth;
    }
  }
  float4 result = (float4)(0.f);
  if (4 * X < C) {
    SET_ON_OR_OFF_VALUE(result.x, N, indices_int[0], on_value, off_value);
  }
  if (4 * X + 1 < C) {
    SET_ON_OR_OFF_VALUE(result.y, N, indices_int[1], on_value, off_value);
  }
  if (4 * X + 2 < C) {
    SET_ON_OR_OFF_VALUE(result.z, N, indices_int[2], on_value, off_value);
  }
  if (4 * X + 3 < C) {
    SET_ON_OR_OFF_VALUE(result.w, N, indices_int[3], on_value, off_value);
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHotAxis1(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int in_index = (N * out_shape.z + Y) * out_shape.w + X;
  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  for (int i = 0; i < C4NUM; i++) {
    if (support_neg_index != 0 && indices_int[i] < 0) {
      indices_int[i] += depth;
    }
  }
  float4 result = (float4)(0.f);
  if (4 * X < C) {
    SET_ON_OR_OFF_VALUE(result.x, H, indices_int[0], on_value, off_value);
  }
  if (4 * X + 1 < C) {
    SET_ON_OR_OFF_VALUE(result.y, H, indices_int[1], on_value, off_value);
  }
  if (4 * X + 2 < C) {
    SET_ON_OR_OFF_VALUE(result.z, H, indices_int[2], on_value, off_value);
  }
  if (4 * X + 3 < C) {
    SET_ON_OR_OFF_VALUE(result.w, H, indices_int[3], on_value, off_value);
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHotAxis2(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int in_index = (N * out_shape.y + H) * out_shape.w + X;
  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  for (int i = 0; i < C4NUM; i++) {
    if (support_neg_index != 0 && indices_int[i] < 0) {
      indices_int[i] += depth;
    }
  }
  float4 result = (float4)(0.f);
  if (4 * X < C) {
    SET_ON_OR_OFF_VALUE(result.x, Y, indices_int[0], on_value, off_value);
  }
  if (4 * X + 1 < C) {
    SET_ON_OR_OFF_VALUE(result.y, Y, indices_int[1], on_value, off_value);
  }
  if (4 * X + 2 < C) {
    SET_ON_OR_OFF_VALUE(result.z, Y, indices_int[2], on_value, off_value);
  }
  if (4 * X + 3 < C) {
    SET_ON_OR_OFF_VALUE(result.w, Y, indices_int[3], on_value, off_value);
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHotAxis3(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int ci4_size = UP_DIV(out_shape.z, C4NUM);
  int in_index_c4 = (N * out_shape.y + H) * ci4_size + Y / 4;
  int in_index_c4_remainder = Y % 4;
  int4 indices =
    READ_IMAGE(src_data, smp_zero, (int2)(in_index_c4 % in_image2d_shape.x, in_index_c4 / in_image2d_shape.x));
  int *indices_int = (int *)&indices;
  int index_one = indices_int[in_index_c4_remainder];
  if (support_neg_index != 0 && index_one < 0) {
    index_one += depth;
  }
  float4 result = (float4)(0.f);
  if (4 * X < C) {
    SET_ON_OR_OFF_VALUE(result.x, 4 * X, index_one, on_value, off_value);
  }
  if (4 * X + 1 < C) {
    SET_ON_OR_OFF_VALUE(result.y, 4 * X + 1, index_one, on_value, off_value);
  }
  if (4 * X + 2 < C) {
    SET_ON_OR_OFF_VALUE(result.z, 4 * X + 2, index_one, on_value, off_value);
  }
  if (4 * X + 3 < C) {
    SET_ON_OR_OFF_VALUE(result.w, 4 * X + 3, index_one, on_value, off_value);
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}

__kernel void OneHot2DAxis3(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,
                            int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W (out_shape.w is 1, Y is always 0)
  int Z = get_global_id(2);  // H * N (out_shape.h is 1, so N == Z)
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  int in_index_c4_remainder = Z % 4;
  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(Z / C4NUM, 0));
  int *indices_int = (int *)&indices;
  int index_one = indices_int[in_index_c4_remainder];
  if (support_neg_index != 0 && index_one < 0) {
    index_one += depth;
  }
  float4 result = (float4)(0.f);
  if (4 * X < C) {
    SET_ON_OR_OFF_VALUE(result.x, 4 * X, index_one, on_value, off_value);
  }
  if (4 * X + 1 < C) {
    SET_ON_OR_OFF_VALUE(result.y, 4 * X + 1, index_one, on_value, off_value);
  }
  if (4 * X + 2 < C) {
    SET_ON_OR_OFF_VALUE(result.z, 4 * X + 2, index_one, on_value, off_value);
  }
  if (4 * X + 3 < C) {
    SET_ON_OR_OFF_VALUE(result.w, 4 * X + 3, index_one, on_value, off_value);
  }
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);
}
