#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define C4NUM 4
__kernel void SparseToDenseScalarDim0(__read_only image2d_t input, __write_only image2d_t output, float weight,
                                      int2 input_shape, float default_value) {
  FLT4 index_input = READ_IMAGE(input, smp_zero, (int2)(0, 0));
  FLT4 result = {default_value, default_value, default_value, default_value};
  int integer = index_input.x / C4NUM;
  int decimal = (int)(index_input.x) % C4NUM;
  if (decimal == 0) {
    result.x = weight;
  } else if (decimal == 1) {
    result.y = weight;
  } else if (decimal == 2) {
    result.z = weight;
  } else {
    result.w = weight;
  }
  WRITE_IMAGE(output, (int2)(0, integer), result);
  return;
}

__kernel void SparseToDenseScalarDim1(__read_only image2d_t input, __write_only image2d_t output, float weight,
                                      int2 input_shape, float default_value) {
  for (int i = 0; i < input_shape.x; ++i) {
    FLT4 result = READ_IMAGE(input, smp_zero, (int2)(0, i));
    int Y = result.x;
    result.x = weight;
    WRITE_IMAGE(output, (int2)(0, Y), result);
  }
}

__kernel void SparseToDenseVectorDim1(__read_only image2d_t input, __write_only image2d_t output,
                                      __global float *weight, int2 input_shape, float default_value) {
  int index_weight = 0;
  for (int i = 0; i < input_shape.x; ++i) {
    FLT4 result = READ_IMAGE(input, smp_zero, (int2)(0, i));
    int Y = result.x;
    result.x = weight[index_weight++];
    WRITE_IMAGE(output, (int2)(0, Y), result);
  }
}

__kernel void SparseToDenseScalarDim2Shape2(__read_only image2d_t input, __write_only image2d_t output, float weight,
                                            int2 input_shape, float default_value) {
  FLT temp[8] = {default_value, default_value, default_value, default_value,
                 default_value, default_value, default_value, default_value};
  FLT result_temp[8] = {default_value, default_value, default_value, default_value,
                        default_value, default_value, default_value, default_value};
  int index = 0;  // 0~4
  int X = 0;
  FLT4 index_begin = READ_IMAGE(input, smp_zero, (int2)(0, 0));
  int Y = (int)index_begin.x;   // N
  temp[index] = index_begin.y;  // c/4
  for (int i = 1; i < input_shape.x && index < C4NUM; ++i) {
    FLT4 index_input = READ_IMAGE(input, smp_zero, (int2)(0, i));
    if ((((int)temp[index]) / C4NUM == ((int)index_input.y) / C4NUM) && (Y == (int)index_input.x)) {
      index++;
      if (index < C4NUM) {
        temp[index] = index_input.y;
      }
    } else {
      for (int j = 0; j <= index && index < C4NUM; ++j) {
        int decimal = (int)temp[j] % C4NUM;
        result_temp[decimal] = weight;
        X = ((int)temp[0]) / C4NUM;
      }
      FLT4 result = {result_temp[0], result_temp[1], result_temp[2], result_temp[3]};
      WRITE_IMAGE(output, (int2)(X, Y), result);
      index = 0;
      Y = (int)index_input.x;
      temp[0] = index_input.y;
      temp[1] = temp[2] = temp[3] = default_value;
      result_temp[0] = result_temp[1] = result_temp[2] = result_temp[3] = default_value;
    }
  }

  // judge the last element for input
  X = ((int)temp[0]) / C4NUM;
  for (int i = 0; i <= index && index < C4NUM; ++i) {
    int decimal = (int)temp[i] % C4NUM;
    result_temp[decimal] = weight;
  }
  FLT4 result = {result_temp[0], result_temp[1], result_temp[2], result_temp[3]};
  WRITE_IMAGE(output, (int2)(X, Y), result);
}

__kernel void SparseToDenseVectorDim2Shape2(__read_only image2d_t input, __write_only image2d_t output,
                                            __global float *weight, int2 input_shape, float default_value) {
  FLT temp[8] = {default_value, default_value, default_value, default_value,
                 default_value, default_value, default_value, default_value};
  FLT result_temp[8] = {default_value, default_value, default_value, default_value,
                        default_value, default_value, default_value, default_value};
  int index = 0;  // 0~4
  int weight_index = 0;
  int X = 0;
  FLT4 index_begin = READ_IMAGE(input, smp_zero, (int2)(0, 0));
  int Y = (int)index_begin.x;   // N
  temp[index] = index_begin.y;  // c/4
  for (int i = 1; i < input_shape.x && index < C4NUM; ++i) {
    FLT4 index_input = READ_IMAGE(input, smp_zero, (int2)(0, i));
    if ((((int)temp[index]) / C4NUM == ((int)index_input.y) / C4NUM) && (Y == (int)index_input.x)) {
      index++;
      if (index < C4NUM) {
        temp[index] = index_input.y;
      }
    } else {
      for (int j = 0; j <= index && index < C4NUM; ++j) {
        int decimal = (int)temp[j] % C4NUM;
        result_temp[decimal] = weight[weight_index++];
        X = ((int)temp[0]) / C4NUM;
      }
      FLT4 result = {result_temp[0], result_temp[1], result_temp[2], result_temp[3]};
      WRITE_IMAGE(output, (int2)(X, Y), result);
      index = 0;
      Y = (int)index_input.x;
      temp[0] = index_input.y;
      temp[1] = temp[2] = temp[3] = default_value;
      result_temp[0] = result_temp[1] = result_temp[2] = result_temp[3] = default_value;
    }
  }

  // judge the last element for input
  X = ((int)temp[0]) / C4NUM;
  for (int i = 0; i <= index && index < C4NUM; ++i) {
    int decimal = (int)temp[i] % C4NUM;
    result_temp[decimal] = weight[weight_index++];
  }
  FLT4 result = {result_temp[0], result_temp[1], result_temp[2], result_temp[3]};
  WRITE_IMAGE(output, (int2)(X, Y), result);
}
