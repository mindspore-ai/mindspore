#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#define C4NUM 4

__kernel void Scale_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,
                        __write_only image2d_t output, const int2 output_shape, const int act_type) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));
  FLT4 s = READ_IMAGE(scale, smp_none, (int2)(X, Y));
  FLT4 o = READ_IMAGE(offset, smp_none, (int2)(X, Y));
  FLT4 out = in * s + o;
  if (act_type == ActivationType_RELU) {
    out = max(out, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));
  }
  WRITE_IMAGE(output, (int2)(X, Y), out);
}

__kernel void BoardcastScale_IMG(__read_only image2d_t input, float scale, float offset, __write_only image2d_t output,
                                 const int2 output_shape, const int act_type) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));
  FLT4 out = in * (FLT)scale + (FLT)offset;
  if (act_type == ActivationType_RELU) {
    out = max(out, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));
  }
  WRITE_IMAGE(output, (int2)(X, Y), out);
}

__kernel void Scale_C_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,
                          __write_only image2d_t output, const int2 output_shape, const int C, const int act_type) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y || C == 0) {
    return;
  }

  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));
  FLT4 s = READ_IMAGE(scale, smp_none, (int2)(X % C, 0));
  FLT4 o = READ_IMAGE(offset, smp_none, (int2)(X % C, 0));
  FLT4 out = in * s + o;
  if (act_type == ActivationType_RELU) {
    out = max(out, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));
  }
  WRITE_IMAGE(output, (int2)(X, Y), out);
}

__kernel void Scale_H_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,
                          __write_only image2d_t output, const int2 output_shape, const int H, const int act_type) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y || H == 0) {
    return;
  }
  int h = Y % H;
  int h_quotient = h / C4NUM;
  int h_remainder = h % C4NUM;
  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));
  FLT4 s = READ_IMAGE(scale, smp_none, (int2)(h_quotient, 0));
  FLT4 o = READ_IMAGE(offset, smp_none, (int2)(h_quotient, 0));
  FLT s_real;
  FLT o_real;
  if (h_remainder == 0) {
    s_real = s.x;
    o_real = o.x;
  } else if (h_remainder == 1) {
    s_real = s.y;
    o_real = o.y;
  } else if (h_remainder == 2) {
    s_real = s.z;
    o_real = o.z;
  } else {
    s_real = s.w;
    o_real = o.w;
  }
  FLT4 out = in * s_real + o_real;
  if (act_type == ActivationType_RELU) {
    out = max(out, (FLT4)(0.0f));
  } else if (act_type == ActivationType_RELU6) {
    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));
  }
  WRITE_IMAGE(output, (int2)(X, Y), out);
}
