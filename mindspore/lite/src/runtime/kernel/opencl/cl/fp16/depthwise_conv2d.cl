#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define ACCUM_FLT4 half4
#define FLT half
#define FLT2 half2
#define FLT3 half3
#define FLT4 half4
#define TO_FLT4 convert_half4
#define TO_ACCUM_TYPE convert_half4
#define TO_ACCUM_FLT convert_half
#define READ_IMAGE read_imagef
#define WRITE_IMAGE write_imagef
__constant sampler_t smp_edge = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void DepthwiseConv2d_NC4HW4(__global FLT4 *src_data, __global FLT4 *filters, __global FLT4 *biases,
                                     float relu_clip1, __global FLT4 *dst_data, int2 kernel_size, int2 stride,
                                     int2 padding, int2 dilation, int4 src_size, int4 dst_size) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  ACCUM_FLT4 r = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int x_offseted = X * stride.x + padding.x;
  int y_offseted = Y * stride.y + padding.y;
  int fx_c = Z * kernel_size.x * kernel_size.y;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = y_offseted + ky * dilation.y;
    bool outside_y = y_c < 0 || y_c >= src_size.y;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = x_offseted + kx * dilation.x;
      bool outside_x = x_c < 0 || x_c >= src_size.x;
      if (!outside_x && !outside_y) {
        FLT4 f = filters[fx_c];
        FLT4 src_final = src_data[(((Z)*src_size.y + (y_c)) * src_size.x + (x_c))];
        r += TO_ACCUM_TYPE(src_final * f);
      }
      fx_c++;
    }
  }
  FLT4 bias_val = biases[Z];
  FLT4 res0 = TO_FLT4(r) + bias_val;
  res0 = clamp(res0, (FLT)(0.0f), (FLT)(relu_clip1));
  dst_data[(((Z)*dst_size.y + (Y)) * dst_size.x + (X))] = res0;
}

__kernel void DepthwiseConv2d_NHWC4(__global FLT4 *src_data, __global FLT4 *filters, __global FLT4 *biases,
                                    float relu_clip1, __global FLT4 *dst_data, int2 kernel_size, int2 stride,
                                    int2 padding, int2 dilation, int4 src_size, int4 dst_size) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;
  ACCUM_FLT4 r = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int x_offseted = X * stride.x + padding.x;
  int y_offseted = Y * stride.y + padding.y;
  int fx_c = Z * kernel_size.x * kernel_size.y;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    int y_c = y_offseted + ky * dilation.y;
    bool outside_y = y_c < 0 || y_c >= src_size.y;
    for (int kx = 0; kx < kernel_size.x; ++kx) {
      int x_c = x_offseted + kx * dilation.x;
      bool outside_x = x_c < 0 || x_c >= src_size.x;
      if (!outside_x && !outside_y) {
        FLT4 f = filters[fx_c];
        FLT4 src_final = src_data[((y_c * src_size.x + x_c) * src_size.z + Z)];
        r += TO_ACCUM_TYPE(src_final * f);
      }
      fx_c++;
    }
  }
  FLT4 bias_val = biases[Z];
  FLT4 res0 = TO_FLT4(r) + bias_val;
  res0 = clamp(res0, (FLT)(0.0f), (FLT)(relu_clip1));
  dst_data[((Y * dst_size.x + X) * dst_size.z + Z)] = res0;
}
