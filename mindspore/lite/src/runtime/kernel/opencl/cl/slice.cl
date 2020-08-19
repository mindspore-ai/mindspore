#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define INT2 int2
#define INT4 int4
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__kernel void slice(__read_only image2d_t input, __write_only image2d_t output, INT4 input_shape, INT4 out_shape,
                    INT4 begin, INT2 sharedNoUpdiv) {
  int X = get_global_id(1);  // H
  int Y = get_global_id(2);  // W
  if (X >= out_shape.y || Y >= out_shape.z) {
    return;
  }
  FLT4 result;
  if (sharedNoUpdiv.x % 4 == 0) {
    for (int i = 0; i < out_shape.w; i++) {
      result = READ_IMAGE(input, smp_none, (INT2)((Y + begin.z) * input_shape.w + (i + begin.w), (X + begin.y)));
      WRITE_IMAGE(output, (INT2)((Y)*out_shape.w + i, (X)), result);
    }
  } else {
    int begin_postion = sharedNoUpdiv.y % 4;
    FLT4 first = READ_IMAGE(input, smp_none, (INT2)((Y + begin.z) * input_shape.w + begin.w, (X + begin.y)));
    if (begin_postion == 1) {
      for (int i = 1; i <= out_shape.w; i++) {
        FLT4 second = READ_IMAGE(input, smp_none, (INT2)((Y + begin.z) * input_shape.w + (begin.w + i), (X + begin.y)));
        result.x = first.y;
        result.y = first.z;
        result.z = first.w;
        result.w = second.x;
        WRITE_IMAGE(output, (INT2)((Y)*out_shape.w + i - 1, (X)), result);
        first.y = second.y;
        first.z = second.z;
        first.w = second.w;
      }
    } else if (begin_postion == 2) {
      for (int i = 1; i <= out_shape.w; i++) {
        FLT4 second = READ_IMAGE(input, smp_none, (INT2)((Y + begin.z) * input_shape.w + (begin.w + i), (X + begin.y)));
        result.x = first.z;
        result.y = first.w;
        result.z = second.x;
        result.w = second.y;
        WRITE_IMAGE(output, (INT2)((Y)*out_shape.w + i - 1, (X)), result);
        first.z = second.z;
        first.w = second.w;
      }
    } else {
      for (int i = 1; i <= out_shape.w; i++) {
        FLT4 second = READ_IMAGE(input, smp_none, (INT2)((Y + begin.z) * input_shape.w + (begin.w + i), (X + begin.y)));
        result.x = first.w;
        result.y = second.x;
        result.z = second.y;
        result.w = second.z;
        WRITE_IMAGE(output, (INT2)((Y)*out_shape.w + i - 1, (X)), result);
        first.w = second.w;
      }
    }
  }
  // judge the line of size
  int size = sharedNoUpdiv.y % 4;
  FLT4 result_fill0;
  if (size == 1) {
    result_fill0.x = result.x;
    result_fill0.y = 0;
    result_fill0.z = 0;
    result_fill0.w = 0;
    WRITE_IMAGE(output, (INT2)((Y)*out_shape.w + out_shape.w - 1, (X)), result_fill0);
  } else if (size == 2) {
    result_fill0.x = result.x;
    result_fill0.y = result.y;
    result_fill0.z = 0;
    result_fill0.w = 0;
    WRITE_IMAGE(output, (INT2)((Y)*out_shape.w + out_shape.w - 1, (X)), result_fill0);
  } else if (size == 3) {
    result_fill0.x = result.x;
    result_fill0.y = result.y;
    result_fill0.z = result.z;
    result_fill0.w = 0;
    WRITE_IMAGE(output, (INT2)((Y)*out_shape.w + out_shape.w - 1, (X)), result_fill0);
  }
}
