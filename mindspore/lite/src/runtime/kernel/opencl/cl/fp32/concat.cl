#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void Concat(__global float *input0, __global float *input1, __global float *output, const int4 input_shape0,
                     const int4 input_shape1, const int4 output_shape, const int axis) {
  uint oh = get_global_id(0);
  uint ow = get_global_id(1);
  uint oc = get_global_id(2);
  uint index_output;
  uint input_idx;
  if ((oh >= output_shape.y || oh < 0) || (ow >= output_shape.z || ow < 0) || (oc >= output_shape.w || oc < 0)) {
    return;
  }
  if (axis == 3) {
    index_output = oh * output_shape.z * output_shape.w + ow * output_shape.w + oc;
    if (oc < input_shape0.w) {
      input_idx = (input_shape0.z * oh + ow) * input_shape0.w + oc;
      output[index_output] = input0[input_idx];
    } else if ((input_shape0.w <= oc) && oc < (input_shape0.w + input_shape1.w)) {
      input_idx = (input_shape1.z * oh + ow) * input_shape1.w + (oc - input_shape0.w);
      output[index_output] = input1[input_idx];
    } else {
      output[index_output] = 0;
    }
  }
}

__kernel void Concat3input(__global float *input0, __global float *input1, __global float *input2,
                           __global float *output, const int4 input_shape0, const int4 input_shape1,
                           const int4 input_shape2, const int4 output_shape, const int axis) {
  uint oh = get_global_id(0);
  uint ow = get_global_id(1);
  uint oc = get_global_id(2);
  uint index_output;
  uint input_idx;
  if ((oh >= output_shape.y || oh < 0) || (ow >= output_shape.z || ow < 0) || (oc >= output_shape.w || oc < 0)) {
    return;
  }
  index_output = oh * output_shape.z * output_shape.w + ow * output_shape.w + oc;
  if (oc < (input_shape0.w + input_shape1.w)) {
    if (oc < input_shape0.w) {
      input_idx = (input_shape0.z * oh + ow) * input_shape0.w + oc;
      output[index_output] = input0[input_idx];
    } else {
      input_idx = (input_shape1.z * oh + ow) * input_shape1.w + (oc - input_shape0.w);
      output[index_output] = input1[input_idx];
    }
  } else {
    if ((input_shape0.w + input_shape1.w + input_shape2.w) <= oc) {
      output[index_output] = 0;
    } else {
      input_idx = (input_shape2.z * oh + ow) * input_shape2.w + (oc - input_shape0.w - input_shape1.w);
      output[index_output] = input2[input_idx];
    }
  }
}
