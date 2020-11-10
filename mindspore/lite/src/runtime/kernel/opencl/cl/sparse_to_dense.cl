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
  int index = 0;
  if (inshapeindex1_dim == 1) {
    index = ((int)index_input.x) * stride_w;
  } else if (inshapeindex1_dim == 2) {
    index = ((int)index_input.x) * stride_w + ((int)index_input.y);
  } else if (inshapeindex1_dim == 3) {
    index = ((int)index_input.x) * stride_w + ((int)index_input.y) * outputshape.w * C4NUM + ((int)index_input.z);
  } else {
    index = ((int)index_input.x) * outputshape.y * stride_w + ((int)index_input.y) * stride_w +
            ((int)index_input.z) * outputshape.w * C4NUM + (int)index_input.w;
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
  int index = 0;
  if (inshapeindex1_dim == 1) {
    index = ((int)index_input.x) * stride_w;
  } else if (inshapeindex1_dim == 2) {
    index = ((int)index_input.x) * stride_w + (int)index_input.y;
  } else if (inshapeindex1_dim == 3) {
    index = ((int)index_input.x) * stride_w + ((int)index_input.y) * outputshape.w * C4NUM + (int)index_input.z;
  } else {
    index = ((int)index_input.x) * outputshape.y * stride_w + ((int)index_input.y) * stride_w +
            ((int)index_input.z) * outputshape.w * C4NUM + (int)index_input.w;
  }
  output[index] = weight_vector[X];
}
