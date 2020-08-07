#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FLT4 float4
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__kernel void Concat(__write_only image2d_t output_image2d, __read_only image2d_t input0_image2d,
                     __read_only image2d_t input1_image2d, int2 shared_int0, int4 shared_out) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int S = 0;
  if (X >= shared_out.y || Y >= shared_out.z) return;
  for (int i = 0; i < shared_int0.x; i++) {
    FLT4 result0 = read_imagef(input0_image2d, smp_none, (int2)((Y)*shared_int0.x + (i), (X)));
    write_imagef(output_image2d, (int2)((Y)*shared_out.w + (S), (X)), result0);
    S++;
  }
  for (int i = 0; i < shared_int0.y; i++) {
    FLT4 result1 = read_imagef(input1_image2d, smp_none, (int2)((Y)*shared_int0.y + (i), (X)));
    write_imagef(output_image2d, (int2)((Y)*shared_out.w + (S), (X)), result1);
    S++;
  }
}

__kernel void Concat3input(__write_only image2d_t output_image2d, __read_only image2d_t input0_image2d,
                           __read_only image2d_t input1_image2d, __read_only image2d_t input2_image2d, int3 shared_int0,
                           int4 shared_out) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  int S = 0;
  if (X >= shared_out.y || Y >= shared_out.z) return;
  for (int i = 0; i < shared_int0.x; i++) {
    FLT4 result0 = read_imagef(input0_image2d, smp_none, (int2)((Y)*shared_int0.x + (i), (X)));
    write_imagef(output_image2d, (int2)((Y)*shared_out.w + (S), (X)), result0);
    S++;
  }
  for (int i = 0; i < shared_int0.y; i++) {
    FLT4 result1 = read_imagef(input1_image2d, smp_none, (int2)((Y)*shared_int0.y + (i), (X)));
    write_imagef(output_image2d, (int2)((Y)*shared_out.w + (S), (X)), result1);
    S++;
  }
  for (int i = 0; i < shared_int0.z; i++) {
    FLT4 result2 = read_imagef(input2_image2d, smp_none, (int2)((Y)*shared_int0.z + (i), (X)));
    write_imagef(output_image2d, (int2)((Y)*shared_out.w + (S), (X)), result2);
    S++;
  }
}
