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

#include "nnacl/fp32_grad/arithmetic_grad.h"
#include <string.h>
#include <math.h>
#include "nnacl/fp32_grad/utils.h"
#include "nnacl/errorcode.h"

void ElementDivNegSquare(const float *nom, const float *denom, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = -nom[i] / (denom[i] * denom[i]);
  }
}

void ElementMulAndDivNegSquare(const float *a, const float *b, const float *denom, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = -a[i] * b[i] / (denom[i] * denom[i]);
  }
}

int ElementAbsGrad(const float *in1, const float *in2, float *out, int element_size) {
  for (int i = 0; i < element_size; i++) {
    out[i] = (in1[i] < 0.f) ? -in2[i] : ((in1[i] > 0.f) ? in2[i] : 0);
  }
  return NNACL_OK;
}

void MaximumByAxes(const float *input0, const float *input1, const float *dy, const int *input0_dims,
                   const int *input1_dims, const int *dy_dims, float *output0, float *output1, int num_dims) {
  int num_output0 = 1;
  int num_output1 = 1;
  bool same_shape = true;
  for (int idx = 0; idx < num_dims; ++idx) {
    num_output0 *= input0_dims[idx];
    num_output1 *= input1_dims[idx];
    if (input0_dims[idx] != input1_dims[idx]) {
      same_shape = false;
    }
  }

  if (same_shape) {
    int input_iter[8] = {0};

    // Iterate through input_data.
    do {
      size_t offset = GetInputOffset(num_dims, input0_dims, input_iter);
      output0[offset] = input0[offset] > input1[offset] ? dy[offset] : 0.;
      output1[offset] = input1[offset] >= input0[offset] ? dy[offset] : 0.;
    } while (NextIndex(num_dims, input0_dims, input_iter));
  } else {
    memset(output0, 0, num_output0 * sizeof(float));  // zero output
    memset(output1, 0, num_output1 * sizeof(float));  // zero output

    int input_iter[8] = {0};
    int axes0[5] = {0};
    int axes1[5] = {0};
    int num_axes0 = 0;
    int num_axes1 = 0;
    for (int i = 0; i < num_dims; i++) {
      if (input0_dims[i] == 1) {
        axes0[num_axes0++] = i;
      }
      if (input1_dims[i] == 1) {
        axes1[num_axes1++] = i;
      }
    }

    do {
      size_t offset0 = GetOutputOffset(num_dims, input0_dims, input_iter, num_axes0, axes0);
      size_t offset1 = GetOutputOffset(num_dims, input1_dims, input_iter, num_axes1, axes1);
      size_t yt_offset = GetInputOffset(num_dims, input0_dims, input_iter);
      output0[offset0] += input0[offset0] > input1[offset1] ? dy[yt_offset] : 0.;
      output1[offset1] += input1[offset1] >= input0[offset0] ? dy[yt_offset] : 0.;
    } while (NextIndex(num_dims, dy_dims, input_iter));
  }
}

void MinimumByAxes(const float *input0, const float *input1, const float *dy, const int *input0_dims,
                   const int *input1_dims, const int *dy_dims, float *output0, float *output1, int num_dims) {
  int num_output0 = 1;
  int num_output1 = 1;
  bool same_shape = true;
  for (int idx = 0; idx < num_dims; ++idx) {
    num_output0 *= input0_dims[idx];
    num_output1 *= input1_dims[idx];
    if (input0_dims[idx] != input1_dims[idx]) {
      same_shape = false;
    }
  }

  if (same_shape) {
    int input_iter[8] = {0};

    // Iterate through input_data.
    do {
      size_t offset = GetInputOffset(num_dims, input0_dims, input_iter);
      output0[offset] = input0[offset] < input1[offset] ? dy[offset] : 0.;
      output1[offset] = input1[offset] <= input0[offset] ? dy[offset] : 0.;
    } while (NextIndex(num_dims, input0_dims, input_iter));
  } else {
    memset(output0, 0, num_output0 * sizeof(float));  // zero output
    memset(output1, 0, num_output1 * sizeof(float));  // zero output

    int input_iter[8] = {0};
    int axes0[5] = {0};
    int axes1[5] = {0};
    int num_axes0 = 0;
    int num_axes1 = 0;
    for (int i = 0; i < num_dims; i++) {
      if (input0_dims[i] == 1) {
        axes0[num_axes0++] = i;
      }
      if (input1_dims[i] == 1) {
        axes1[num_axes1++] = i;
      }
    }

    do {
      size_t offset0 = GetOutputOffset(num_dims, input0_dims, input_iter, num_axes0, axes0);
      size_t offset1 = GetOutputOffset(num_dims, input1_dims, input_iter, num_axes1, axes1);
      size_t yt_offset = GetInputOffset(num_dims, input0_dims, input_iter);
      output0[offset0] += input0[offset0] < input1[offset1] ? dy[yt_offset] : 0.;
      output1[offset1] += input1[offset1] <= input0[offset0] ? dy[yt_offset] : 0.;
    } while (NextIndex(num_dims, dy_dims, input_iter));
  }
}

int ElementSqrtGrad(const float *in1, const float *in2, float *out, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    out[i] = 0.5f * in2[i] / in1[i];
  }
  return NNACL_OK;
}

int ElementRsqrtGrad(const float *in1, const float *in2, float *out, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    out[i] = -0.5f * in2[i] * in1[i] * in1[1] * in1[i];
  }
  return NNACL_OK;
}
