#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C4NUM 4
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void SparseToDenseScalar(__read_only image2d_t input, __global float *output, float weight, int2 inputshape,
                                  int4 outputshape, float default_value, int stride_w, int inshapeindex1_dim) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= inputshape.x || Y >= inputshape.y) {
    return;
  }
  FLT4 index_input = READ_IMAGE(input, smp_zero, (int2)(Y, X));
  int4 index_input_int = *((int4 *)&index_input);
  int index = 0;
  if (inshapeindex1_dim == 1) {
    index = (index_input_int.x) * stride_w;
  } else if (inshapeindex1_dim == 2) {
    index = (index_input_int.x) * stride_w + (index_input_int.y);
  } else if (inshapeindex1_dim == 3) {
    index = (index_input_int.x) * stride_w + (index_input_int.y) * outputshape.w * C4NUM + (index_input_int.z);
  } else {
    index = (index_input_int.x) * outputshape.y * stride_w + (index_input_int.y) * stride_w +
            (index_input_int.z) * outputshape.w * C4NUM + index_input_int.w;
  }
  output[index] = weight;
}

__kernel void SparseToDenseVector(__read_only image2d_t input, __global float *output, __global float *weight_vector,
                                  int2 inputshape, int4 outputshape, float default_value, int stride_w,
                                  int inshapeindex1_dim) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= inputshape.x || Y >= inputshape.y) {
    return;
  }
  FLT4 index_input = READ_IMAGE(input, smp_zero, (int2)(Y, X));
  int4 index_input_int = *((int4 *)&index_input);
  int index = 0;
  if (inshapeindex1_dim == 1) {
    index = (index_input_int.x) * stride_w;
  } else if (inshapeindex1_dim == 2) {
    index = (index_input_int.x) * stride_w + index_input_int.y;
  } else if (inshapeindex1_dim == 3) {
    index = (index_input_int.x) * stride_w + (index_input_int.y) * outputshape.w * C4NUM + index_input_int.z;
  } else {
    index = (index_input_int.x) * outputshape.y * stride_w + (index_input_int.y) * stride_w +
            (index_input_int.z) * outputshape.w * C4NUM + index_input_int.w;
  }
  output[index] = weight_vector[X];
}
