#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void reshape_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 src_size,
                            int4 dst_size) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int CO4 = UP_DIV(dst_size.z, C4NUM);
  int CO4_rem = dst_size.z % C4NUM;
  if (X >= dst_size.x || Y > dst_size.y) {
    return;
  }
  int CI4 = UP_DIV(src_size.x, C4NUM);
  int CI4_rem = src_size.x % C4NUM;
  CI4_rem = (CI4_rem == 0) ? C4NUM : CI4_rem;
  int in_img_x = CI4 * src_size.y;
  DTYPE4 res = (DTYPE4)(0.0f);
  DTYPE tmp[4];
  DTYPE res_tmp[4];
  int gcnt = 0;
  if (CO4_rem == 0 && ((CI4_rem & 0x3) == 0)) {
    gcnt = X + dst_size.x * Y;
    res = READ_IMAGE(src_data, smp_zero, (int2)(gcnt % in_img_x, gcnt / in_img_x));
    WRITE_IMAGE(dst_data, (int2)(X, Y), res);
  } else {
    int start = ((X / CO4 * dst_size.z + min(dst_size.z, (X % CO4) * C4NUM)) + dst_size.w * Y);
    gcnt = start / src_size.x * CI4 + (start % src_size.x) / C4NUM;
    start = start % src_size.x % C4NUM;
    for (int i = 0, n = 0, j = start; i < C4NUM; ++n, j = 0) {
      int X_src = (gcnt + n) % in_img_x;
      res = READ_IMAGE(src_data, smp_zero, (int2)(X_src, (gcnt + n) / in_img_x));
      tmp[0] = res.x;
      tmp[1] = res.y;
      tmp[2] = res.z;
      tmp[3] = res.w;
      int k = (X_src % CI4) == (CI4 - 1) ? CI4_rem : C4NUM;
      for (; j < k && i < C4NUM; ++j, ++i) {
        res_tmp[i] = tmp[j];
      }
    }
    res.x = res_tmp[0];
    res.y = res_tmp[1];
    res.z = res_tmp[2];
    res.w = res_tmp[3];
    WRITE_IMAGE(dst_data, (int2)(X, Y), res);
  }
}
