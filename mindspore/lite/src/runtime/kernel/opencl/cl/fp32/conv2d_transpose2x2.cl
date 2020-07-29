#define FLT float
#define FLT4 float4
#define FLT16 float16
__kernel void conv2d_transpose2x2(__global FLT4 *inputx, __global FLT16 *weight, __global FLT4 *bias,
                                  __global FLT4 *output, int2 kernel_size, int2 stride, int2 padding, int4 src_size,
                                  int4 dst_size) {
  int h = get_global_id(0);
  int w = get_global_id(1);
  int co = get_global_id(2);
  if (h * 2 >= dst_size.x || w * 2 >= dst_size.y || co >= dst_size.z) return;
  FLT4 r0 = (FLT4)(0.f);
  FLT4 r1 = (FLT4)(0.f);
  FLT4 r2 = (FLT4)(0.f);
  FLT4 r3 = (FLT4)(0.f);
  int base_x = (h * src_size.y + w) * src_size.z;
  int base_w = co * src_size.z;
  for (int ci = 0; ci < src_size.z; ++ci) {
    FLT4 x = inputx[base_x + ci];
    FLT16 w0 = weight[(base_w + ci) * 4];
    FLT16 w1 = weight[(base_w + ci) * 4 + 1];
    FLT16 w2 = weight[(base_w + ci) * 4 + 2];
    FLT16 w3 = weight[(base_w + ci) * 4 + 3];
    r0 += x.x * w0.s0123;
    r0 += x.y * w0.s4567;
    r0 += x.z * w0.s89ab;
    r0 += x.w * w0.scdef;

    r1 += x.x * w1.s0123;
    r1 += x.y * w1.s4567;
    r1 += x.z * w1.s89ab;
    r1 += x.w * w1.scdef;

    r2 += x.x * w2.s0123;
    r2 += x.y * w2.s4567;
    r2 += x.z * w2.s89ab;
    r2 += x.w * w2.scdef;

    r3 += x.x * w3.s0123;
    r3 += x.y * w3.s4567;
    r3 += x.z * w3.s89ab;
    r3 += x.w * w3.scdef;
  }
  r0 += bias[co];
  r1 += bias[co];
  r2 += bias[co];
  r3 += bias[co];
  output[((2 * h + 0) * dst_size.y + 2 * w + 0) * dst_size.z + co] = r0;
  output[((2 * h + 0) * dst_size.y + 2 * w + 1) * dst_size.z + co] = r1;
  output[((2 * h + 1) * dst_size.y + 2 * w + 0) * dst_size.z + co] = r2;
  output[((2 * h + 1) * dst_size.y + 2 * w + 1) * dst_size.z + co] = r3;
}