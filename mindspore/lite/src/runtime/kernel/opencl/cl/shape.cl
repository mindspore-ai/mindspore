#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void Shape(__read_only int4 input, __write_only image2d_t output,
                        ) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  printf("X:%d,Y:%d",X,Y);
  WRITE_IMAGE(output, (int2)(0, 0), input);
}
