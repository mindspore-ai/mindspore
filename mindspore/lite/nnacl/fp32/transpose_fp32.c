/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/fp32/transpose_fp32.h"

void TransposeDim2Fp32(const float *in_data, float *out_data, const int *strides, int *out_strides, const int *perm,
                       const int *output_shape) {
  const int stride0 = strides[perm[0]];
  const int stride1 = strides[perm[1]];
  const int output0 = output_shape[0];
  const int output1 = output_shape[1];
  for (int i = 0; i < output0; ++i) {
    int out_stride0_i = i * output1;
    int stride0_i = i * 1 * stride0;
    for (int j = 0; j < output1; ++j) {
      out_data[out_stride0_i + j] = in_data[stride0_i + j * stride1];
    }
  }
}

void TransposeDim3Fp32(const float *in_data, float *out_data, const int *strides, const int *out_strides,
                       const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[0]];
  const int stride1 = strides[perm[1]];
  const int stride2 = strides[perm[2]];
  const int out_stride0 = out_strides[0];
  const int out_stride1 = out_strides[1];
  const int output0 = output_shape[0];
  const int output1 = output_shape[1];
  const int output2 = output_shape[2];
  for (int i = 0; i < output0; ++i) {
    int out_stride0_i = i * out_stride0;
    int stride0_i = i * stride0;
    for (int j = 0; j < output1; ++j) {
      int out_stride1_j = j * out_stride1;
      int stride1_j = j * stride1;
      for (int k = 0; k < output2; ++k) {
        out_data[out_stride0_i + out_stride1_j + k] = in_data[stride0_i + stride1_j + k * stride2];
      }
    }
  }
}

void TransposeDim4Fp32(const float *in_data, float *out_data, const int *strides, const int *out_strides,
                       const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[0]];
  const int stride1 = strides[perm[1]];
  const int stride2 = strides[perm[2]];
  const int stride3 = strides[perm[3]];
  const int out_stride0 = out_strides[0];
  const int out_stride1 = out_strides[1];
  const int out_stride2 = out_strides[2];
  const int output0 = output_shape[0];
  const int output1 = output_shape[1];
  const int output2 = output_shape[2];
  const int output3 = output_shape[3];

  for (int i = 0; i < output0; ++i) {
    int out_stride0_i = i * out_stride0;
    int stride0_i = i * stride0;
    for (int j = 0; j < output1; ++j) {
      int out_stride1_j = j * out_stride1;
      int stride1_j = j * stride1;
      for (int k = 0; k < output2; ++k) {
        int out_stride2_k = k * out_stride2;
        int stride2_k = k * stride2;
        for (int m = 0; m < output3; ++m) {
          out_data[out_stride0_i + out_stride1_j + out_stride2_k + m] =
            in_data[stride0_i + stride1_j + stride2_k + m * stride3];
        }
      }
    }
  }
}

void TransposeDim5Fp32(const float *in_data, float *out_data, const int *strides, const int *out_strides,
                       const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[0]];
  const int stride1 = strides[perm[1]];
  const int stride2 = strides[perm[2]];
  const int stride3 = strides[perm[3]];
  const int stride4 = strides[perm[4]];
  const int out_stride0 = out_strides[0];
  const int out_stride1 = out_strides[1];
  const int out_stride2 = out_strides[2];
  const int out_stride3 = out_strides[3];
  const int output0 = output_shape[0];
  const int output1 = output_shape[1];
  const int output2 = output_shape[2];
  const int output3 = output_shape[3];
  const int output4 = output_shape[4];

  for (int i = 0; i < output0; ++i) {
    int out_stride0_i = i * out_stride0;
    int stride0_i = i * stride0;
    for (int j = 0; j < output1; ++j) {
      int out_stride1_j = j * out_stride1;
      int stride1_j = j * stride1;
      for (int k = 0; k < output2; ++k) {
        int out_stride2_k = k * out_stride2;
        int stride2_k = k * stride2;
        for (int m = 0; m < output3; ++m) {
          int out_stride3_m = m * out_stride3;
          int stride3_m = m * stride3;
          for (int n = 0; n < output4; ++n) {
            out_data[out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + n] =
              in_data[stride0_i + stride1_j + stride2_k + stride3_m + n * stride4];
          }
        }
      }
    }
  }
}

void TransposeDim6Fp32(const float *in_data, float *out_data, const int *strides, const int *out_strides,
                       const int *perm, const int *output_shape) {
  const int stride0 = strides[perm[0]];
  const int stride1 = strides[perm[1]];
  const int stride2 = strides[perm[2]];
  const int stride3 = strides[perm[3]];
  const int stride4 = strides[perm[4]];
  const int stride5 = strides[perm[5]];
  const int out_stride0 = out_strides[0];
  const int out_stride1 = out_strides[1];
  const int out_stride2 = out_strides[2];
  const int out_stride3 = out_strides[3];
  const int out_stride4 = out_strides[4];
  const int output0 = output_shape[0];
  const int output1 = output_shape[1];
  const int output2 = output_shape[2];
  const int output3 = output_shape[3];
  const int output4 = output_shape[4];
  const int output5 = output_shape[5];

  for (int i = 0; i < output0; ++i) {
    int out_stride0_i = i * out_stride0;
    int stride0_i = i * stride0;
    for (int j = 0; j < output1; ++j) {
      int out_stride1_j = j * out_stride1;
      int stride1_j = j * stride1;
      for (int k = 0; k < output2; ++k) {
        int out_stride2_k = k * out_stride2;
        int stride2_k = k * stride2;
        for (int m = 0; m < output3; ++m) {
          int out_stride3_m = m * out_stride3;
          int stride3_m = m * stride3;
          for (int n = 0; n < output4; ++n) {
            int out_stride4_m = n * out_stride4;
            int stride4_m = n * stride4;
            for (int g = 0; g < output5; ++g) {
              out_data[out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + out_stride4_m + g] =
                in_data[stride0_i + stride1_j + stride2_k + stride3_m + stride4_m + g * stride5];
            }
          }
        }
      }
    }
  }
}

void TransposeDimsFp32(const float *in_data, float *out_data, const int *output_shape, int *size, int *position,
                       TransposeParameter *transpose_param, int task_id, int thread_num) {
  int *perm = transpose_param->perm_;
  int *strides = transpose_param->strides_;
  int *out_strides = transpose_param->out_strides_;
  int num_axes = transpose_param->num_axes_;
  size_t data_size = (*size) * output_shape[0];
  size_t offset_size = UP_DIV(data_size, thread_num);
  size_t task_offset = offset_size * task_id;
  int count = data_size - task_offset;
  if (count <= 0) {
    return;
  }
  count = MSMIN(offset_size, count);
  for (size_t idx = task_offset; idx < task_offset + count; ++idx) {
    int pos = idx;
    int output_idx = 0;
    int input_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      *(position + i) = pos / *(size + i);
      int out_stride = i < num_axes - 1 ? out_strides[i] : 1;
      output_idx += (*(position + i) * out_stride);
      input_idx += (*(position + i) * strides[perm[i]]);
      pos -= *(position + i) * (*(size + i));
    }
    out_data[output_idx] = in_data[input_idx];
  }
}

int DoTransposeFp32(const float *in_data, float *out_data, const int *output_shape,
                    TransposeParameter *transpose_param) {
  if (in_data == NULL || out_data == NULL) {
    return NNACL_ERR;
  }
  int *perm = transpose_param->perm_;
  int *strides = transpose_param->strides_;
  int *out_strides = transpose_param->out_strides_;
  int data_size = transpose_param->data_size_;
  int num_axes = transpose_param->num_axes_;

  if (num_axes < 2) {
    return NNACL_ERR;
  }

  // check if transpose is needed
  bool needTranspose = false;
  for (int i = 1; i < num_axes; ++i) {
    if (perm[i] - perm[i - 1] != 1) {
      needTranspose = true;
      break;
    }
  }

  if (!needTranspose) {
    (void)memcpy(out_data, in_data, data_size);
    return NNACL_OK;
  }
  for (int i = 0; i < num_axes; ++i) {
    if (perm[i] < 0) {
      return NNACL_PARAM_INVALID;
    }
  }
  if (num_axes == 2) {
    TransposeDim2Fp32(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == 3) {
    TransposeDim3Fp32(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == 4) {
    TransposeDim4Fp32(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == 5) {
    TransposeDim5Fp32(in_data, out_data, strides, out_strides, perm, output_shape);
  } else if (num_axes == 6) {
    TransposeDim6Fp32(in_data, out_data, strides, out_strides, perm, output_shape);
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}
