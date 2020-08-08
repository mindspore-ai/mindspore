#define FLT half
#define FLT4 half4
#define FLT16 half16
#define READ_IMAGE read_imageh
#define WRITE_IMAGE write_imageh
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void conv2d_transpose2x2(__read_only image2d_t src_data, __global FLT16 *weight, __read_only image2d_t biases,
                                  __write_only image2d_t dst_data, int2 kernel_size, int2 stride, int2 padding,
                                  int4 src_size, int4 dst_size) {
  int h = get_global_id(0);
  int kh = h % 2;
  int src_h = h / 2;
  src_h = src_h * 2;
  int w = get_global_id(1);
  int kw = w % 2;
  int src_w = w / 2;
  src_w = src_w * 2;
  int co = get_global_id(2);
  if (src_h * 2 >= dst_size.x || src_w * 2 >= dst_size.y || co >= dst_size.z) return;
  FLT4 r0 = (FLT4)(0.f);
  FLT4 r1 = (FLT4)(0.f);
  FLT4 r2 = (FLT4)(0.f);
  FLT4 r3 = (FLT4)(0.f);
  int base_w = (co * 4 + kh + kw * 2) * src_size.z;
  for (int ci = 0; ci < src_size.z; ++ci) {
    FLT4 x0 = READ_IMAGE(src_data, smp_zero, (int2)(src_w * src_size.z + ci, src_h));
    FLT4 x1 = READ_IMAGE(src_data, smp_zero, (int2)(src_w * src_size.z + ci, src_h + 1));
    FLT4 x2 = READ_IMAGE(src_data, smp_zero, (int2)((src_w + 1) * src_size.z + ci, src_h));
    FLT4 x3 = READ_IMAGE(src_data, smp_zero, (int2)((src_w + 1) * src_size.z + ci, src_h + 1));
    FLT16 weight_cache = weight[base_w++];
    r0 += x0.x * weight_cache.s0123;
    r0 += x0.y * weight_cache.s4567;
    r0 += x0.z * weight_cache.s89ab;
    r0 += x0.w * weight_cache.scdef;

    r1 += x1.x * weight_cache.s0123;
    r1 += x1.y * weight_cache.s4567;
    r1 += x1.z * weight_cache.s89ab;
    r1 += x1.w * weight_cache.scdef;

    r2 += x2.x * weight_cache.s0123;
    r2 += x2.y * weight_cache.s4567;
    r2 += x2.z * weight_cache.s89ab;
    r2 += x2.w * weight_cache.scdef;

    r3 += x3.x * weight_cache.s0123;
    r3 += x3.y * weight_cache.s4567;
    r3 += x3.z * weight_cache.s89ab;
    r3 += x3.w * weight_cache.scdef;
  }
  FLT4 bias_val = READ_IMAGE(biases, smp_zero, (int2)(co, 0));
  r0 += bias_val;
  r1 += bias_val;
  r2 += bias_val;
  r3 += bias_val;

  WRITE_IMAGE(dst_data, (int2)((2 * src_w + kw) * dst_size.z + co, 2 * src_h + kh), r0);
  WRITE_IMAGE(dst_data, (int2)((2 * src_w + kw) * dst_size.z + co, 2 * src_h + kh + 2), r1);
  WRITE_IMAGE(dst_data, (int2)((2 * src_w + kw + 2) * dst_size.z + co, 2 * src_h + kh), r2);
  WRITE_IMAGE(dst_data, (int2)((2 * src_w + kw + 2) * dst_size.z + co, 2 * src_h + kh + 2), r3);
}
