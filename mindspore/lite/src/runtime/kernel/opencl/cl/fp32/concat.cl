//#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void Concat(__global float *input0, __global float *input1, __global float *output, const int4 input_shape0,
                     const int4 input_shape1, const int4 output_shape, const int axis) {
  int postion = 0, index_input_shape0 = 0, index_input_shape1 = 0;
  switch (axis) {
    case 1:
      for (int i = 0; i < output_shape.x; i++) {
        for (int j = 0; j < output_shape.y; j++) {
          for (int k = 0; k < output_shape.z; k++) {
            for (int w = 0; w < output_shape.w; w++) {
              postion = i * output_shape.y * output_shape.z * output_shape.w + j * output_shape.z * output_shape.w +
                        k * output_shape.w + w;
              if (j < input_shape0.y) {
                output[postion] = input0[index_input_shape0++];
              } else {
                output[postion] = input1[index_input_shape1++];
              }
            }
          }
        }
      }
      break;
    case 2:
      for (int i = 0; i < output_shape.x; i++) {
        for (int j = 0; j < output_shape.y; j++) {
          for (int k = 0; k < output_shape.z; k++) {
            for (int w = 0; w < output_shape.w; w++) {
              postion = i * output_shape.y * output_shape.z * output_shape.w + j * output_shape.z * output_shape.w +
                        k * output_shape.w + w;
              if (k < input_shape0.z) {
                output[postion] = input0[index_input_shape0++];
              } else {
                output[postion] = input1[index_input_shape1++];
              }
            }
          }
        }
      }
      break;
    case 3:
      for (int i = 0; i < output_shape.x; i++) {
        for (int j = 0; j < output_shape.y; j++) {
          for (int k = 0; k < output_shape.z; k++) {
            for (int w = 0; w < output_shape.w; w++) {
              postion = i * output_shape.y * output_shape.z * output_shape.w + j * output_shape.z * output_shape.w +
                        k * output_shape.w + w;
              if (w < input_shape0.w) {
                output[postion] = input0[index_input_shape0++];
              } else {
                output[postion] = input1[index_input_shape1++];
              }
            }
          }
        }
      }
      break;
    default:
      break;
  }
}