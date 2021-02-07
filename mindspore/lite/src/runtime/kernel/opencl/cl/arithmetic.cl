#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAdd(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementSub(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementMul(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementDiv(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                         const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementLogicalAnd(__read_only image2d_t input_a, __read_only image2d_t input_b,
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

__kernel void ElementLogicalOr(__read_only image2d_t input_a, __read_only image2d_t input_b,
                               __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementMaximum(__read_only image2d_t input_a, __read_only image2d_t input_b,
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

__kernel void ElementMinimum(__read_only image2d_t input_a, __read_only image2d_t input_b,
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

__kernel void ElementFloorDiv(__read_only image2d_t input_a, __read_only image2d_t input_b,
                              __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementFloorMod(__read_only image2d_t input_a, __read_only image2d_t input_b,
                              __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  FLT4 result = a - floor(divide_no_check(a, b)) * b;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void ElementSquaredDifference(__read_only image2d_t input_a, __read_only image2d_t input_b,
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

__kernel void ElementEqual(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                           const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementNotEqual(__read_only image2d_t input_a, __read_only image2d_t input_b,
                              __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementLess(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                          const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementLessEqual(__read_only image2d_t input_a, __read_only image2d_t input_b,
                               __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {
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

__kernel void ElementGreater(__read_only image2d_t input_a, __read_only image2d_t input_b,
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

__kernel void ElementGreaterEqual(__read_only image2d_t input_a, __read_only image2d_t input_b,
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

__kernel void BroadcastNHWC4Add(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // N * H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {
    return;
  }
  int H = Z % output_shape.y;
  int N = Z / output_shape.y;
  int a_c = X < a_shape.w ? X : 0;
  int a_w = Y < a_shape.z ? Y : 0;
  int a_h = H < a_shape.y ? H : 0;
  int a_n = N < a_shape.x ? N : 0;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));
  int b_c = X < b_shape.w ? X : 0;
  int b_w = Y < b_shape.z ? Y : 0;
  int b_h = H < b_shape.y ? H : 0;
  int b_n = N < b_shape.x ? N : 0;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));
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

__kernel void BroadcastNHWC4BiasAdd(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                    __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                    const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // N * H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {
    return;
  }
  int H = Z % output_shape.y;
  int N = Z / output_shape.y;
  int a_c = X < a_shape.w ? X : 0;
  int a_w = Y < a_shape.z ? Y : 0;
  int a_h = H < a_shape.y ? H : 0;
  int a_n = N < a_shape.x ? N : 0;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));
  int b_c = X < b_shape.w ? X : 0;
  int b_w = Y < b_shape.z ? Y : 0;
  int b_h = H < b_shape.y ? H : 0;
  int b_n = N < b_shape.x ? N : 0;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));
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

__kernel void BroadcastNHWC4Sub(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // N * H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {
    return;
  }
  int H = Z % output_shape.y;
  int N = Z / output_shape.y;
  int a_c = X < a_shape.w ? X : 0;
  int a_w = Y < a_shape.z ? Y : 0;
  int a_h = H < a_shape.y ? H : 0;
  int a_n = N < a_shape.x ? N : 0;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));
  int b_c = X < b_shape.w ? X : 0;
  int b_w = Y < b_shape.z ? Y : 0;
  int b_h = H < b_shape.y ? H : 0;
  int b_n = N < b_shape.x ? N : 0;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));
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

__kernel void BroadcastNHWC4Mul(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // N * H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {
    return;
  }
  int H = Z % output_shape.y;
  int N = Z / output_shape.y;
  int a_c = X < a_shape.w ? X : 0;
  int a_w = Y < a_shape.z ? Y : 0;
  int a_h = H < a_shape.y ? H : 0;
  int a_n = N < a_shape.x ? N : 0;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));
  int b_c = X < b_shape.w ? X : 0;
  int b_w = Y < b_shape.z ? Y : 0;
  int b_h = H < b_shape.y ? H : 0;
  int b_n = N < b_shape.x ? N : 0;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));
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

__kernel void BroadcastNHWC4Div(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // N * H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {
    return;
  }
  int H = Z % output_shape.y;
  int N = Z / output_shape.y;
  int a_c = X < a_shape.w ? X : 0;
  int a_w = Y < a_shape.z ? Y : 0;
  int a_h = H < a_shape.y ? H : 0;
  int a_n = N < a_shape.x ? N : 0;
  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));
  int b_c = X < b_shape.w ? X : 0;
  int b_w = Y < b_shape.z ? Y : 0;
  int b_h = H < b_shape.y ? H : 0;
  int b_n = N < b_shape.x ? N : 0;
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));
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

__kernel void BroadcastLogicalAnd(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastLogicalOr(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastMaximum(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastMinimum(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastFloorDiv(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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
__kernel void BroadcastNHWC4FloorMod(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                     __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                     const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(Y * a_shape.w + X, Z));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, 0));
  FLT4 result = a - floor(divide_no_check(a, b)) * b;
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);
}

__kernel void BroadcastNHWC4SquaredDifference(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                              __write_only image2d_t output, const int4 a_shape, const int4 b_shape,
                                              const int4 output_shape, const int broadcastC_flag, float act_min,
                                              float act_max) {
  int X = get_global_id(0);  // C4
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // H
  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(Y * a_shape.w + X, Z));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, 0));
  FLT4 result = pown((a - b), (int4)2);
  result = clamp(result, (FLT)(act_min), (FLT)(act_max));
  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);
}

__kernel void BroadcastEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastNotEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastLess(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastLessEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastGreater(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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

__kernel void BroadcastGreaterEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,
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
