#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define C4NUM 4

__kernel void ComputeMeanVarAxis3NHWC4(__read_only image2d_t src_data, __global FLT *mean_, __global FLT *variance_,
                                       int4 in_shape, int normalized_shape_size) {
  int X = get_global_id(0);  // n*h
  int Y = get_global_id(1);  // w
  if (X > in_shape.x * in_shape.y || Y > in_shape.z || in_shape.y == 0 || normalized_shape_size == 0) {
    return;
  }
  int n = X / in_shape.y;
  int h = X % in_shape.y;
  int w = Y;
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int remainder = in_shape.w % C4NUM;
  FLT4 mean_temp = {0.0f, 0.0f, 0.0f, 0.0f};
  FLT4 var_temp = {0.0f, 0.0f, 0.0f, 0.0f};
  FLT mean = 0.0f;
  FLT var = 0.0f;

  // compute mean
  for (int i = 0; i < ci4; ++i) {
    FLT4 result_temp = READ_IMAGE(src_data, smp_none, (int2)(w * ci4 + i, n * in_shape.y + h));
    mean_temp += result_temp;
  }
  mean = (mean_temp.x + mean_temp.y + mean_temp.z + mean_temp.w) / normalized_shape_size;
  mean_temp.x = mean_temp.y = mean_temp.z = mean_temp.w = mean;

  // compute var
  for (int i = 0; i < ci4; ++i) {
    FLT4 result_temp = READ_IMAGE(src_data, smp_none, (int2)(w * ci4 + i, n * in_shape.y + h));
    if ((i + 1) * C4NUM <= in_shape.w) {
      var_temp += (result_temp - mean_temp) * (result_temp - mean_temp);
    } else {
      if (remainder == 1) {
        mean_temp.x = mean;
        mean_temp.y = mean_temp.z = mean_temp.w = 0.0f;
      } else if (remainder == 2) {
        mean_temp.x = mean_temp.y = mean;
        mean_temp.z = mean_temp.w = 0.0f;
      } else {
        mean_temp.x = mean_temp.y = mean_temp.z = mean;
        mean_temp.w = 0.0f;
      }
      var_temp += (result_temp - mean_temp) * (result_temp - mean_temp);
    }
  }
  var = (var_temp.x + var_temp.y + var_temp.z + var_temp.w) / normalized_shape_size;

  // write result to dst
  int position = (n * in_shape.y + h) * in_shape.z + w;
  mean_[position] = mean;
  variance_[position] = var;
}

__kernel void LayerNormalization_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data,
                                       __global FLT *mean_, __global FLT *variance_, __global FLT *gamma_,
                                       __global FLT *beta_, int4 in_shape, float epsilon_, int begin_params_axis_) {
  int X = get_global_id(0);  // n*h
  int Y = get_global_id(1);  // w
  int Z = get_global_id(2);  // c4
  if (X >= in_shape.x * in_shape.y || Y >= in_shape.z || Z >= in_shape.w || in_shape.y == 0) {
    return;
  }
  int n = X / in_shape.y;
  int h = X % in_shape.y;
  int w = Y;
  int c = Z;
  int ci4 = UP_DIV(in_shape.w, C4NUM);
  int postion_mv = 0;
  int postion_gb = 0;
  if (begin_params_axis_ == 1) {
    postion_mv = n;
    postion_gb = (h * in_shape.z + w) * ci4 * C4NUM + c * C4NUM;
  } else if (begin_params_axis_ == 2) {
    postion_mv = n * in_shape.y + h;
    postion_gb = w * ci4 * C4NUM + c * C4NUM;
  } else if (begin_params_axis_ == 3) {
    postion_mv = (n * in_shape.y + h) * in_shape.z + w;
    postion_gb = c * C4NUM;
  }
  FLT4 result = {0.0f, 0.0f, 0.0f, 0.0f};
  FLT4 result_in = READ_IMAGE(src_data, smp_none, (int2)(w * ci4 + c, n * in_shape.y + h));
  result.x = ((result_in.x - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb] +
             beta_[postion_gb];
  result.y = ((result_in.y - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb + 1] +
             beta_[postion_gb + 1];
  result.z = ((result_in.z - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb + 2] +
             beta_[postion_gb + 2];
  result.w = ((result_in.w - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb + 3] +
             beta_[postion_gb + 3];
  WRITE_IMAGE(dst_data, (int2)((w * ci4 + c), (n * in_shape.y + h)), result);
}
