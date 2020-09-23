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
#include "internal/src/kernel/fp32/bias_add.h"
#include "internal/src/kernel/common/common_infershape.h"
#include "internal/include/model.h"
#include "internal/include/ms_tensor.h"
#include "internal/include/lite_utils.h"
#include "internal/src/lite_log.h"
#include "internal/include/errorcode.h"
#include "nnacl/arithmetic_common.h"
#include "nnacl/fp32/arithmetic.h"

int DoBiasAddInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param) {
  return DoCommonInferShape(in_tensors, out_tensors);
}

int DoBiasAdd(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
              mindspore::lite::Allocator *allocator) {
  if (in_tensors.size() != 2 || in_tensors[0]->data_ == NULL || in_tensors[1]->data_ == NULL) {
    LITE_LOG_ERROR("input tensors num not correct or input data is NULL!");
    return RET_INPUT_TENSOR_ERROR;
  }
  if (out_tensors.size() != 1 || out_tensors[0]->data_ == NULL) {
    LITE_LOG_ERROR("output tensors num not correct or output data is NULL!");
    return RET_ERROR;
  }
  if (allocator == NULL) {
    LITE_LOG_ERROR("allocator is NULL!");
    return RET_ERROR;
  }
  ArithmeticParameter *params = reinterpret_cast<ArithmeticParameter *>(node->primitive_);

  ShapeVector dims = in_tensors[0]->shape_;
  params->ndim_ = dims.size();
  for (size_t i = 0; i < params->ndim_; i++) {
    params->in_shape0_[i] = dims[i];
    params->in_shape1_[i] = 1;
    params->out_shape_[i] = dims[i];
  }
  params->in_shape1_[params->ndim_ - 1] = dims[params->ndim_ - 1];

  float *in = reinterpret_cast<float *>(in_tensors[0]->data_);
  float *bias = reinterpret_cast<float *>(in_tensors[1]->data_);
  float *out = reinterpret_cast<float *>(out_tensors[0]->data_);
  size_t data_size = in_tensors[0]->ElementsNum();
  float *tile_in = reinterpret_cast<float *>(allocator->Malloc(data_size * sizeof(float)));
  float *tile_bias = reinterpret_cast<float *>(allocator->Malloc(data_size * sizeof(float)));
  if (tile_in == NULL || tile_bias == NULL) {
    LITE_LOG_ERROR("Memory allocation failed!");
    allocator->Free(tile_in);
    allocator->Free(tile_bias);
    return RET_ERROR;
  }
  BroadcastAdd(in, bias, tile_in, tile_bias, out, data_size, params);
  allocator->Free(tile_in);
  allocator->Free(tile_bias);
  return RET_OK;
}
