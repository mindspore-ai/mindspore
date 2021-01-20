#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define C4NUM 4
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void SpaceToDepth(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,
                           int4 out_shape, int block_size, int ci_size) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  if (out_shape.y == 0 || ci_size == 0 || block_size == 0) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int co_base = X * C4NUM;
  FLT result[C4NUM] = {0.f};
  for (int i = 0; i < C4NUM; i++) {
    int co = co_base + i;
    int ci = co % ci_size;
    int hw_block = co / ci_size;
    int hi = H * block_size + hw_block / block_size;
    int wi = Y * block_size + hw_block % block_size;
    int ci4 = ci / C4NUM;
    int ci4_ramainder = ci % C4NUM;
    FLT4 tmp = READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci4, N * in_shape.y + hi));
    if (ci4_ramainder == 0) {
      result[i] = tmp.x;
    } else if (ci4_ramainder == 1) {
      result[i] = tmp.y;
    } else if (ci4_ramainder == 2) {
      result[i] = tmp.z;
    } else {
      result[i] = tmp.w;
    }
  }
  FLT4 result_flt4 = {result[0], result[1], result[2], result[3]};
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result_flt4);
}

__kernel void SpaceToDepthAlign(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,
                                int4 out_shape, int block_size, int ci_size) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  if (out_shape.y == 0 || in_shape.w == 0 || block_size == 0) return;

  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int ni = N;
  int ci = X % in_shape.w;
  int hw_block = X / in_shape.w;
  int hi = H * block_size + hw_block / block_size;
  int wi = Y * block_size + hw_block % block_size;
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z),
              READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci, ni * in_shape.y + hi)));
}

__kernel void DepthToSpace(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,
                           int4 out_shape, int block_size, int co_size) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  if (out_shape.y == 0 || block_size == 0) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int co_base = X * C4NUM;
  FLT result[C4NUM] = {0.f};
  for (int i = 0; i < C4NUM; i++) {
    int co = co_base + i;
    int bh = H % block_size;
    int hi = H / block_size;
    int bw = Y % block_size;
    int wi = Y / block_size;
    int ci = (bh * block_size + bw) * co_size + co;
    int ci4 = ci / C4NUM;
    int ci4_ramainder = ci % C4NUM;
    FLT4 tmp = READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci4, N * in_shape.y + hi));
    if (ci4_ramainder == 0) {
      result[i] = tmp.x;
    } else if (ci4_ramainder == 1) {
      result[i] = tmp.y;
    } else if (ci4_ramainder == 2) {
      result[i] = tmp.z;
    } else {
      result[i] = tmp.w;
    }
  }
  FLT4 result_flt4 = {result[0], result[1], result[2], result[3]};
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result_flt4);
}

__kernel void DepthToSpaceAlign(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,
                                int4 out_shape, int block_size, int co_size) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H * N
  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;
  if (out_shape.y == 0 || block_size == 0) return;
  int N = Z / out_shape.y;
  int H = Z % out_shape.y;
  int ni = N;
  int bh = H % block_size;
  int hi = H / block_size;
  int bw = Y % block_size;
  int wi = Y / block_size;
  int ci = (bh * block_size + bw) * out_shape.w + X;
  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z),
              READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci, ni * in_shape.y + hi)));
}
