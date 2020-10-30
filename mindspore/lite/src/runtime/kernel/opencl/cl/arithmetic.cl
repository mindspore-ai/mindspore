#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAdd_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a + b;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementSub_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a - b;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementMul_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a * b;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementDiv_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = divide_no_check(a, b);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementAnd_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = AS_FLT4(AS_UINT4(a) & AS_UINT4(b));
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementOr_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                            const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = AS_FLT4(AS_UINT4(a) | AS_UINT4(b));
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementMax_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = max(a, b);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementMin_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = min(a, b);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementFloorDiv_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                  __write_only image2d_t output, const int2 output_shape, float act_min,
                                  float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = floor(divide_no_check(a, b));
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementFloorMod_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                  __write_only image2d_t output, const int2 output_shape, float act_min,
                                  float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = floor(divide_no_check(a, b)) * b;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementSquaredDifference_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                           __write_only image2d_t output, const int2 output_shape, float act_min,
                                           float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = pown((a - b), (int4)2);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                               __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a == b ? (FLT4)1.f : (FLT4).0f;
  // error?
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementNotEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                  __write_only image2d_t output, const int2 output_shape, float act_min,
                                  float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a != b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementLess_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                              __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a < b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementLessEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                   __write_only image2d_t output, const int2 output_shape, float act_min,
                                   float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a <= b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementGreater_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                 __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a > b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementGreaterEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                      __write_only image2d_t output, const int2 output_shape, float act_min,
                                      float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a >= b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastNHWC4Add_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                    __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                    const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y) {
    return;
  }
  int a_c = X < a_shape.w ? X : a_shape.w - 1;
  int a_w = Y < a_shape.z ? Y : a_shape.z - 1;
  int a_h = Z < a_shape.y ? Z : a_shape.y - 1;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_h));
  int b_c = X < b_shape.w ? X : b_shape.w - 1;
  int b_w = Y < b_shape.z ? Y : b_shape.z - 1;
  int b_h = Z < b_shape.y ? Z : b_shape.y - 1;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_h));
  FLT4 result;
  if (broadcastC_flag == 0) {
    result = a + b;
  } else if (broadcastC_flag == 1) {
    result = a.x + b;
  } else {
    result = a + b.x;
  }
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);
}

__kernel void BroadcastNHWC4Sub_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                    __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                    const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y) {
    return;
  }
  int a_c = X < a_shape.w ? X : a_shape.w - 1;
  int a_w = Y < a_shape.z ? Y : a_shape.z - 1;
  int a_h = Z < a_shape.y ? Z : a_shape.y - 1;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_h));
  int b_c = X < b_shape.w ? X : b_shape.w - 1;
  int b_w = Y < b_shape.z ? Y : b_shape.z - 1;
  int b_h = Z < b_shape.y ? Z : b_shape.y - 1;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_h));
  FLT4 result;
  if (broadcastC_flag == 0) {
    result = a - b;
  } else if (broadcastC_flag == 1) {
    result = a.x - b;
  } else {
    result = a - b.x;
  }
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);
}

__kernel void BroadcastNHWC4Mul_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                    __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                    const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y) {
    return;
  }
  int a_c = X < a_shape.w ? X : a_shape.w - 1;
  int a_w = Y < a_shape.z ? Y : a_shape.z - 1;
  int a_h = Z < a_shape.y ? Z : a_shape.y - 1;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_h));
  int b_c = X < b_shape.w ? X : b_shape.w - 1;
  int b_w = Y < b_shape.z ? Y : b_shape.z - 1;
  int b_h = Z < b_shape.y ? Z : b_shape.y - 1;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_h));
  FLT4 result;
  if (broadcastC_flag == 0) {
    result = a * b;
  } else if (broadcastC_flag == 1) {
    result = a.x * b;
  } else {
    result = a * b.x;
  }
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);
}

__kernel void BroadcastNHWC4Div_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                    __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                    const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y) {
    return;
  }
  int a_c = X < a_shape.w ? X : a_shape.w - 1;
  int a_w = Y < a_shape.z ? Y : a_shape.z - 1;
  int a_h = Z < a_shape.y ? Z : a_shape.y - 1;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_h));
  int b_c = X < b_shape.w ? X : b_shape.w - 1;
  int b_w = Y < b_shape.z ? Y : b_shape.z - 1;
  int b_h = Z < b_shape.y ? Z : b_shape.y - 1;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_h));
  FLT4 result;
  if (broadcastC_flag == 0) {
    result = a / b;
  } else if (broadcastC_flag == 1) {
    result = a.x / b;
  } else {
    result = a / b.x;
  }
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);
}

__kernel void BroadcastAnd_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = AS_FLT4(AS_UINT4(a) & (UINT4)((FLT)b));
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastOr_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                              const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = AS_FLT4(AS_UINT4(a) | (UINT4)((FLT)b));
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastMax_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = max(a, (FLT4)b);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastMin_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = min(a, (FLT4)b);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastFloorDiv_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                    const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = floor(divide_no_check(a, (FLT4)b));
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastFloorMod_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                    const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = floor(divide_no_check(a, (FLT4)b)) * (FLT)b;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastSquaredDifference_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                             const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = pown((a - (FLT4)b), (int4)2);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                 const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = a == (FLT4)b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastNotEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                    const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = a != (FLT4)b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastLess_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = a < (FLT4)b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastLessEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                     const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = a <= (FLT4)b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastGreater_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                   const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = a > (FLT4)b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void BroadcastGreaterEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                        const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 result = a >= (FLT4)b ? (FLT4)1.f : (FLT4).0f;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementAdd_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] + input_b[idx];
}

__kernel void ElementSub_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] - input_b[idx];
}

__kernel void ElementMul_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b[idx];
}

__kernel void ElementDiv_BUF(__global float *input_a, __global float *input_b, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b[idx];
}

__kernel void BroadcastAdd_BUF(__global float *input_a, float b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] + (FLT)b;
}

__kernel void BroadcastSub_BUF(__global float *input_a, float b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] - (FLT)b;
}

__kernel void BroadcastMul_BUF(__global float *input_a, float b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * (FLT)b;
}

__kernel void BroadcastDiv_BUF(__global float *input_a, float b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = divide_no_check(input_a[idx], (FLT)b);
}
