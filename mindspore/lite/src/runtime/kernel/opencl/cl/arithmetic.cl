#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define divide_no_check(a, b) (a / b)
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ElementAdd_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a + b);
}

__kernel void ElementSub_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a - b);
}

__kernel void ElementMul_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a * b);
}

__kernel void ElementDiv_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), divide_no_check(a, b));
}

__kernel void ElementAnd_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(AS_UINT4(a) & AS_UINT4(b)));
}

__kernel void ElementOr_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,
                            const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(AS_UINT4(a) | AS_UINT4(b)));
}

__kernel void ElementMax_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), max(a, b));
}

__kernel void ElementMin_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                             __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), min(a, b));
}

__kernel void ElementFloorDiv_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                  __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), floor(a / b));
}

__kernel void ElementFloorMod_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                  __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), floor(divide_no_check(a, b)) * b);
}

__kernel void ElementSquaredDifference_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                           __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), pown((a - b), (int4)2));
}

__kernel void ElementEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                               __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a == b));
}

__kernel void ElementNotEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                  __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a != b));
}

__kernel void ElementLess_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                              __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a < b));
}

__kernel void ElementLessEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                   __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a <= b));
}

__kernel void ElementGreater_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                 __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a > b));
}

__kernel void ElementGreaterEqual_IMG(__read_only image2d_t input_a, __read_only image2d_t input_b,
                                      __write_only image2d_t output, const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a >= b));
}

__kernel void BroadcastAdd_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a + (FLT)b);
}

__kernel void BroadcastSub_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a - (FLT)b);
}

__kernel void BroadcastMul_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), a * (FLT)b);
}

__kernel void BroadcastDiv_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), divide_no_check(a, (FLT)b));
}
__kernel void BroadcastAnd_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(AS_UINT4(a) & (UINT4)((FLT)b)));
}

__kernel void BroadcastOr_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                              const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(AS_UINT4(a) | (UINT4)((FLT)b)));
}

__kernel void BroadcastMax_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), max(a, (FLT4)b));
}

__kernel void BroadcastMin_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                               const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), min(a, (FLT4)b));
}

__kernel void BroadcastFloorDiv_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                    const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), floor(a / (FLT4)b));
}

__kernel void BroadcastFloorMod_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                    const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), floor(divide_no_check(a, (FLT4)b)) * (FLT)b);
}

__kernel void BroadcastSquaredDifference_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                             const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), pown((a - (FLT4)b), (int4)2));
}

__kernel void BroadcastEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                 const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a == (FLT4)b));
}

__kernel void BroadcastNotEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                    const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a != (FLT4)b));
}

__kernel void BroadcastLess_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a < (FLT4)b));
}

__kernel void BroadcastLessEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                     const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a <= (FLT4)b));
}

__kernel void BroadcastGreater_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                   const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a > (FLT4)b));
}

__kernel void BroadcastGreaterEqual_IMG(__read_only image2d_t input_a, float b, __write_only image2d_t output,
                                        const int2 output_shape) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= output_shape.x || Y >= output_shape.y) {
    return;
  }

  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));
  WRITE_IMAGE(output, (int2)(X, Y), AS_FLT4(a >= (FLT4)b));
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
