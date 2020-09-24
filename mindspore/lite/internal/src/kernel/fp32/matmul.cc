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

#include "internal/src/kernel/fp32/matmul.h"
#include "nnacl/fp32/matmul.h"
#include "internal/include/errorcode.h"
#include "internal/include/ms_tensor.h"
#include "internal/src/lite_log.h"

typedef struct MatMulCPUKernelData {
  float *a_c12_ptr_;
  float *b_r8_ptr_;
  float *bias_ptr_;
} MatMulCPUKernelData;

void MatMulInitMatrixA(float *src_ptr, float *dst_ptr, MatMulParameter *params) {
  for (int i = 0; i < params->batch; i++) {
    float *src = src_ptr + i * params->deep_ * params->row_;
    float *dst = dst_ptr + i * params->deep_ * params->row_12_;
    if (params->a_transpose_) {
      RowMajor2Row12Major(src, dst, params->deep_, params->row_);
    } else {
      RowMajor2Col12Major(src, dst, params->row_, params->deep_);
    }
  }
}

void MatMulInitMatrixB(float *src_ptr, float *dst_ptr, MatMulParameter *params) {
  for (int i = 0; i < params->batch; i++) {
    float *src = src_ptr + i * params->deep_ * params->col_;
    float *dst = dst_ptr + i * params->deep_ * params->col_8_;
    if (params->b_transpose_) {
      RowMajor2Col8Major(src, dst, params->col_, params->deep_);
    } else {
      RowMajor2Row8Major(src, dst, params->deep_, params->col_);
    }
  }
}

void FreeMatMulKernelData(MatMulCPUKernelData *kernel_data, mindspore::lite::Allocator *allocator) {
  if (kernel_data == NULL) {
    return;
  }
  if (kernel_data->a_c12_ptr_ != NULL) {
    allocator->Free(kernel_data->a_c12_ptr_);
    kernel_data->a_c12_ptr_ = NULL;
  }

  if (kernel_data->b_r8_ptr_ != NULL) {
    allocator->Free(kernel_data->b_r8_ptr_);
    kernel_data->b_r8_ptr_ = NULL;
  }

  if (kernel_data->bias_ptr_ != NULL) {
    allocator->Free(kernel_data->bias_ptr_);
    kernel_data->bias_ptr_ = NULL;
  }
  free(kernel_data);
}

int DoMatMulInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param) {
  TensorPtr input0 = in_tensors.at(0);
  MS_ASSERT(input0 != nullptr);
  TensorPtr input1 = in_tensors.at(1);
  MS_ASSERT(input1 != nullptr);
  TensorPtr output = out_tensors.at(0);
  MS_ASSERT(output != nullptr);

  int in_datatype[2] = {input0->data_type_, input1->data_type_};
  int in_format[2] = {static_cast<int>(input0->format_), static_cast<int>(input1->format_)};
  size_t dim_size[2] = {input0->shape_.size(), input1->shape_.size()};
  int *in_shape[2] = {input0->shape_.data(), input1->shape_.data()};
  int out_format;
  int out_datatype;
  output->shape_.resize(input0->shape_.size());
  int ret = MatMulInferShape(in_shape, 2, dim_size, output->shape_.data(), in_format, &out_format, in_datatype,
                             &out_datatype, param);
  if (ret != NNACL_OK) {
    LITE_ERROR_LOG("matmul infershape fail!ret: %d", ret);
    return RET_ERROR;
  }
  output->format_ = static_cast<Format>(out_format);
  output->data_type_ = static_cast<TypeId>(out_datatype);
  return RET_OK;
}

int DoMatMul(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
             mindspore::lite::Allocator *allocator) {
  if (in_tensors[0]->data_ == NULL || in_tensors[1]->data_ == NULL) {
    LITE_LOG_ERROR("input data is NULL!");
    return RET_PARAM_INVALID;
  }
  if (allocator == NULL) {
    LITE_LOG_ERROR("input allocator is NULL!");
    return RET_PARAM_INVALID;
  }
  int batch = 1;
  ShapeVector a_shape = in_tensors[0]->shape_;
  ShapeVector c_shape = out_tensors[0]->shape_;
  if (in_tensors.size() == 3) {
    ShapeVector bias_shape = in_tensors[2]->shape_;
    if (bias_shape[bias_shape.size() - 1] != c_shape[c_shape.size() - 1]) {
      LITE_ERROR_LOG("The bias' dimension %d is not equal with column %d", bias_shape[bias_shape.size() - 1],
                     c_shape[c_shape.size() - 1]);
      return RET_INPUT_TENSOR_ERROR;
    }
  }
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    batch *= a_shape[i];
  }

  MatMulParameter *params = (MatMulParameter *)node->primitive_;
  params->batch = batch;
  params->row_ = c_shape[c_shape.size() - 2];
  params->col_ = c_shape[c_shape.size() - 1];
  params->deep_ = params->a_transpose_ ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
  params->row_12_ = UP_ROUND(params->row_, C12NUM);
  params->col_8_ = UP_ROUND(params->col_, 8);

  MatMulCPUKernelData *kernel_data = (MatMulCPUKernelData *)malloc(sizeof(MatMulCPUKernelData));
  if (kernel_data == NULL) {
    LITE_LOG_ERROR("Malloc MatMulCPUKernelData failed");
    return RET_MEMORY_FAILED;
  }
  kernel_data->a_c12_ptr_ =
    reinterpret_cast<float *>(allocator->Malloc(params->batch * params->row_12_ * params->deep_ * sizeof(float)));
  if (kernel_data->a_c12_ptr_ == NULL) {
    FreeMatMulKernelData(kernel_data, allocator);
    return RET_MEMORY_FAILED;
  }
  memset(kernel_data->a_c12_ptr_, 0, params->row_12_ * params->deep_ * sizeof(float));

  kernel_data->b_r8_ptr_ =
    reinterpret_cast<float *>(allocator->Malloc(params->batch * params->col_8_ * params->deep_ * sizeof(float)));
  if (kernel_data->b_r8_ptr_ == NULL) {
    FreeMatMulKernelData(kernel_data, allocator);
    return RET_MEMORY_FAILED;
  }
  memset(kernel_data->b_r8_ptr_, 0, params->col_8_ * params->deep_ * sizeof(float));

  MatMulInitMatrixA((float *)in_tensors[0]->data_, kernel_data->a_c12_ptr_, params);
  MatMulInitMatrixB((float *)in_tensors[1]->data_, kernel_data->b_r8_ptr_, params);
  kernel_data->bias_ptr_ = (float *)(allocator->Malloc(params->col_8_ * sizeof(float)));
  if (kernel_data->bias_ptr_ == NULL) {
    FreeMatMulKernelData(kernel_data, allocator);
    return RET_MEMORY_FAILED;
  }
  memset(kernel_data->bias_ptr_, 0, params->col_8_ * sizeof(float));

  if (in_tensors.size() == 3) {
    memcpy(kernel_data->bias_ptr_, in_tensors[2]->data_, params->col_ * sizeof(float));
  }
  auto c_src = (float *)out_tensors[0]->data_;
  for (int i = 0; i < params->batch; ++i) {
    float *a_ptr = kernel_data->a_c12_ptr_ + i * params->row_12_ * params->deep_;
    float *b_ptr = kernel_data->b_r8_ptr_ + i * params->deep_ * params->col_8_;
    float *c_ptr = c_src + i * params->row_ * params->col_;
    MatMulOpt(a_ptr, b_ptr, c_ptr, kernel_data->bias_ptr_, ActType_No, params->deep_, params->row_, params->col_,
              params->col_, OutType_Nhwc);
  }

  return RET_OK;
}
