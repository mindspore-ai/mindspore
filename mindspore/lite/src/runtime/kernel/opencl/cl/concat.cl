#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void Concat(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                     int4 input_shape0, int4 input_shape1, int4 output_shape, const int axis) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  if (axis == 0) {
    if (X < input_shape0.x * input_shape0.y) {
      FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    } else {
      FLT4 result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.x * input_shape0.y)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    }
  } else if (axis == 1) {
    if (X < input_shape0.y) {
      FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    } else {
      FLT4 result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.y)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    }
  } else if (axis == 2) {
    if (Y < input_shape0.z) {
      FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    } else {
      FLT4 result = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z) * input_shape1.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    }
  } else {
    if (Z < input_shape0.w) {
      FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    } else {
      FLT4 result = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z - input_shape0.w, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
    }
  }
}

__kernel void Concat3input(__read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,
                           __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2,
                           int4 output_shape, const int axis) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  if (axis == 0) {
    if (X < input_shape0.x * input_shape0.y) {
      FLT4 result0 = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result0);
    } else if (X < (input_shape0.x * input_shape0.y + input_shape1.x * input_shape1.y)) {
      FLT4 result1 =
        READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.x * input_shape0.y)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result1);
    } else {
      FLT4 result2 = READ_IMAGE(
        input2, smp_none,
        (int2)((Y)*input_shape2.w + Z, (X - input_shape0.x * input_shape0.y - input_shape1.x * input_shape1.y)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result2);
    }
  } else if (axis == 1) {
    if (X < input_shape0.y) {
      FLT4 result0 = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result0);
    } else if (X < (input_shape0.y + input_shape1.y)) {
      FLT4 result1 = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z, (X - input_shape0.y)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result1);
    } else {
      FLT4 result2 =
        READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z, (X - input_shape0.y - input_shape1.y)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result2);
    }
  } else if (axis == 2) {
    if (Y < input_shape0.z) {
      FLT4 result0 = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result0);
    } else if (Y < (input_shape0.z + input_shape0.z)) {
      FLT4 result1 = READ_IMAGE(input1, smp_none, (int2)((Y - input_shape0.z) * input_shape1.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result1);
    } else {
      FLT4 result2 =
        READ_IMAGE(input2, smp_none, (int2)((Y - input_shape0.z - input_shape1.z) * input_shape2.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result2);
    }
  } else {
    if (Z < input_shape0.w) {
      FLT4 result0 = READ_IMAGE(input0, smp_none, (int2)((Y)*input_shape0.w + Z, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result0);
    } else if (Z < (input_shape0.w + input_shape0.w)) {
      FLT4 result1 = READ_IMAGE(input1, smp_none, (int2)((Y)*input_shape1.w + Z - input_shape0.w, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result1);
    } else {
      FLT4 result2 =
        READ_IMAGE(input2, smp_none, (int2)((Y)*input_shape2.w + Z - input_shape0.w - input_shape1.w, (X)));
      WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result2);
    }
  }
}
