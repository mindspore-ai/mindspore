#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void transpose_0312_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {
  int X = get_global_id(0);  // H4, C4 for src
  int Y = get_global_id(1);  // W, H for src
  int Z = get_global_id(2);  // C4, W4 for src
  if (4 * X >= shape.y || Y >= shape.z || 4 * Z >= shape.w) {
    return;
  }
  int H4 = UP_DIV(shape.y, 4);
  int C4 = UP_DIV(shape.w, 4);
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z * H4 + X, Y));
  FLT4 src1 = (FLT4)0.f;
  if (4 * Z + 1 < shape.w) {
    src1 = READ_IMAGE(src_data, smp_zero, (int2)((4 * Z + 1) * H4 + X, Y));
  }
  FLT4 src2 = (FLT4)0.f;
  if (4 * Z + 2 < shape.w) {
    src2 = READ_IMAGE(src_data, smp_zero, (int2)((4 * Z + 2) * H4 + X, Y));
  }
  FLT4 src3 = (FLT4)0.f;
  if (4 * Z + 3 < shape.w) {
    src3 = READ_IMAGE(src_data, smp_zero, (int2)((4 * Z + 3) * H4 + X, Y));
  }
  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);
  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);
  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);
  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);
  WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X), dst0);
  if (4 * X + 1 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 1), dst1);
  }
  if (4 * X + 2 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 2), dst2);
  }
  if (4 * X + 3 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 3), dst3);
  }
}

__kernel void transpose_0312_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {
  int X = get_global_id(0);  // H4, C4 for src
  int Y = get_global_id(1);  // W, H for src
  int Z = get_global_id(2);  // C4, W4 for src
  if (4 * X >= shape.y || Y >= shape.z || 4 * Z >= shape.w) {
    return;
  }
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z, X * shape.z + Y));
  FLT4 src1 = (FLT4)0.f;
  if (4 * Z + 1 < shape.w) {
    src1 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z + 1, X * shape.z + Y));
  }
  FLT4 src2 = (FLT4)0.f;
  if (4 * Z + 2 < shape.w) {
    src2 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z + 2, X * shape.z + Y));
  }
  FLT4 src3 = (FLT4)0.f;
  if (4 * Z + 3 < shape.w) {
    src3 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z + 3, X * shape.z + Y));
  }
  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);
  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);
  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);
  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);
  WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X), dst0);
  if (4 * X + 1 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X + 1), dst1);
  }
  if (4 * X + 2 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X + 2), dst2);
  }
  if (4 * X + 3 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X + 3), dst3);
  }
}

__kernel void transpose_0312_oversize_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data,
                                            int4 shape) {
  int X = get_global_id(0);  // H4, C4 for src
  int Y = get_global_id(1);  // W, H for src
  int Z = get_global_id(2);  // C4, W4 for src
  if (4 * X >= shape.y || Y >= shape.z || 4 * Z >= shape.w) {
    return;
  }
  int H4 = UP_DIV(shape.y, 4);
  int C4 = UP_DIV(shape.w, 4);
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z));
  FLT4 src1 = (FLT4)0.f;
  if (4 * Z + 1 < shape.w) {
    src1 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 1));
  }
  FLT4 src2 = (FLT4)0.f;
  if (4 * Z + 2 < shape.w) {
    src2 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 2));
  }
  FLT4 src3 = (FLT4)0.f;
  if (4 * Z + 3 < shape.w) {
    src3 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 3));
  }
  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);
  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);
  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);
  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);
  WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X), dst0);
  if (4 * X + 1 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 1), dst1);
  }
  if (4 * X + 2 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 2), dst2);
  }
  if (4 * X + 3 < shape.y) {
    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 3), dst3);
  }
}

__kernel void transpose_0231_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {
  int X = get_global_id(0);  // H, W for src
  int Y = get_global_id(1);  // W4, C4 for src
  int Z = get_global_id(2);  // C4, H4 for src
  if (X >= shape.y || 4 * Y >= shape.z || 4 * Z >= shape.w) {
    return;
  }
  int W4 = UP_DIV(shape.z, 4);
  int C4 = UP_DIV(shape.w, 4);
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z));
  FLT4 src1 = (FLT4)0.f;
  if (4 * Z + 1 < shape.w) {
    src1 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z + 1));
  }
  FLT4 src2 = (FLT4)0.f;
  if (4 * Z + 2 < shape.w) {
    src2 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z + 2));
  }
  FLT4 src3 = (FLT4)0.f;
  if (4 * Z + 3 < shape.w) {
    src3 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z + 3));
  }
  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);
  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);
  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);
  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);
  WRITE_IMAGE(dst_data, (int2)(4 * Y * C4 + Z, X), dst0);
  if (4 * Y + 1 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)((4 * Y + 1) * C4 + Z, X), dst1);
  }
  if (4 * Y + 2 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)((4 * Y + 2) * C4 + Z, X), dst2);
  }
  if (4 * Y + 3 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)((4 * Y + 3) * C4 + Z, X), dst3);
  }
}

__kernel void transpose_0231_oversize_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data,
                                            int4 shape) {
  int X = get_global_id(0);  // H, W for src
  int Y = get_global_id(1);  // W4, C4 for src
  int Z = get_global_id(2);  // C4, H4 for src
  if (X >= shape.y || 4 * Y >= shape.z || 4 * Z >= shape.w) {
    return;
  }
  int W4 = UP_DIV(shape.z, 4);
  int C4 = UP_DIV(shape.w, 4);
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * Z * shape.y + X));
  FLT4 src1 = (FLT4)0.f;
  if (4 * Z + 1 < shape.w) {
    src1 = READ_IMAGE(src_data, smp_zero, (int2)(Y, (4 * Z + 1) * shape.y + X));
  }
  FLT4 src2 = (FLT4)0.f;
  if (4 * Z + 2 < shape.w) {
    src2 = READ_IMAGE(src_data, smp_zero, (int2)(Y, (4 * Z + 2) * shape.y + X));
  }
  FLT4 src3 = (FLT4)0.f;
  if (4 * Z + 3 < shape.w) {
    src3 = READ_IMAGE(src_data, smp_zero, (int2)(Y, (4 * Z + 3) * shape.y + X));
  }
  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);
  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);
  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);
  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);
  WRITE_IMAGE(dst_data, (int2)(4 * Y * C4 + Z, X), dst0);
  if (4 * Y + 1 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)((4 * Y + 1) * C4 + Z, X), dst1);
  }
  if (4 * Y + 2 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)((4 * Y + 2) * C4 + Z, X), dst2);
  }
  if (4 * Y + 3 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)((4 * Y + 3) * C4 + Z, X), dst3);
  }
}

__kernel void transpose_0231_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {
  int X = get_global_id(0);  // H, W for src
  int Y = get_global_id(1);  // W4, C4 for src
  int Z = get_global_id(2);  // C4, H4 for src
  if (X >= shape.y || 4 * Y >= shape.z || 4 * Z >= shape.w) {
    return;
  }
  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z));
  FLT4 src1 = (FLT4)0.f;
  if (4 * Z + 1 < shape.w) {
    src1 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 1));
  }
  FLT4 src2 = (FLT4)0.f;
  if (4 * Z + 2 < shape.w) {
    src2 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 2));
  }
  FLT4 src3 = (FLT4)0.f;
  if (4 * Z + 3 < shape.w) {
    src3 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 3));
  }
  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);
  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);
  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);
  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);
  WRITE_IMAGE(dst_data, (int2)(4 * Y, Z * shape.y + X), dst0);
  if (4 * Y + 1 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)(4 * Y + 1, Z * shape.y + X), dst1);
  }
  if (4 * Y + 2 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)(4 * Y + 2, Z * shape.y + X), dst2);
  }
  if (4 * Y + 3 < shape.z) {
    WRITE_IMAGE(dst_data, (int2)(4 * Y + 3, Z * shape.y + X), dst3);
  }
}

typedef union FLT4_array {
  FLT c_array[4];
  FLT4 vector;
} FLT4_array;

__kernel void transpose_general_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 out_shape,
                                      int4 de_perm, int4 in_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= out_shape.y * out_shape.x || Y >= out_shape.z || 4 * Z >= out_shape.w) {
    return;
  }
  int N = X / out_shape.y;
  int H = X % out_shape.y;
  int CI4_SIZE = UP_DIV(in_shape.w, 4);
  FLT4_array result_tmp;
  result_tmp.vector = (FLT4)(0.f);
  FLT *result_ptr = result_tmp.c_array;
  for (int i = 0; i < 4; i++) {
    if (Z * 4 + i < out_shape.w) {
      int out_index[4] = {N, H, Y, Z * 4 + i};
      FLT4 src = READ_IMAGE(src_data, smp_zero,
                            (int2)(out_index[de_perm.z] * CI4_SIZE + out_index[de_perm.w] / 4,
                                   out_index[de_perm.x] * in_shape.y + out_index[de_perm.y]));
      FLT4_array src_tmp;
      src_tmp.vector = src;
      result_tmp.c_array[i] = src_tmp.c_array[out_index[de_perm.w] % 4];
    }
  }
  int CO4_SIZE = UP_DIV(out_shape.w, 4);
  WRITE_IMAGE(dst_data, (int2)(Y * CO4_SIZE + Z, X), result_tmp.vector);
}
