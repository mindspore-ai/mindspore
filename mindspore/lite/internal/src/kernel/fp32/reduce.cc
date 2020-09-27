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

#include "internal/src/kernel/fp32/reduce.h"
#include "internal/include/model.h"
#include "internal/include/lite_utils.h"
#include "internal/src/lite_log.h"
#include "internal/include/errorcode.h"
#include "nnacl/reduce_parameter.h"
#include "nnacl/fp32/reduce.h"
#include "nnacl/errorcode.h"

typedef int (*Reducer)(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                       float *dst_data, const int tid, const int thread_num);

int MallocTmpBuffer(float *data_buffers[], const ShapeVector &shape, const int *axes, const int num_axes,
                    mindspore::lite::Allocator *allocator) {
  ShapeVector input_shape = shape;
  const int rank = input_shape.size();
  for (auto i = 0; i < num_axes - 1; i++) {
    int axis = axes[i];
    size_t size = 1;
    for (int j = 0; j < rank; j++) {
      if (axis != j) {
        size *= input_shape[j];
      }
    }
    float *buffer = reinterpret_cast<float *>(allocator->Malloc(size * sizeof(float)));
    if (buffer == nullptr) {
      LITE_LOG_ERROR("Memory allocation failed!");
      return RET_ERROR;
    }
    data_buffers[i] = buffer;
    input_shape[axis] = 1;
  }
  return RET_OK;
}

void FreeTmpBuffer(float *data_buffers[], int size, mindspore::lite::Allocator *allocator) {
  if (data_buffers == nullptr) {
    return;
  }
  for (int i = 0; i < size; ++i) {
    allocator->Free(data_buffers[i]);
    data_buffers[i] = nullptr;
  }
}

int RunReduce(Reducer reducer, float *data_buffers[], float *in_data, float *out_data, ReduceParameter *params,
              ShapeVector shape) {
  int rank = shape.size();
  float *dst_data = NULL;
  float *src_data = in_data;
  ShapeVector tmp_shape = shape;
  for (int i = 0; i < params->num_axes_; ++i) {
    if (i != params->num_axes_ - 1) {
      dst_data = data_buffers[i];
    } else {
      dst_data = out_data;
    }
    int axis = params->axes_[i];
    int outer_size = 1;
    for (int j = 0; j < axis; j++) {
      outer_size *= tmp_shape[j];
    }
    int inner_size = 1;
    for (int k = axis + 1; k < rank; k++) {
      inner_size *= tmp_shape[k];
    }
    int axis_size = tmp_shape[axis];
    int error_code = reducer(outer_size, inner_size, axis_size, src_data, dst_data, 0, 1);
    if (error_code != RET_OK) {
      LITE_LOG_ERROR("Reduce run error!");
      return RET_ERROR;
    }
    tmp_shape[axis] = 1;
    src_data = dst_data;
  }
  return RET_OK;
}

int DoReduceInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param) {
  if (in_tensors.size() != 1 || in_tensors[0]->data_ == NULL) {
    LITE_LOG_ERROR("input tensors num not correct or input data is NULL!");
    return RET_INPUT_TENSOR_ERROR;
  }
  if (out_tensors.size() != 1) {
    LITE_LOG_ERROR("output tensors num not correct!");
    return RET_ERROR;
  }

  int in_datatype[1] = {in_tensors[0]->data_type_};
  int in_format[1] = {static_cast<int>(in_tensors[0]->format_)};
  size_t dim_size[1] = {in_tensors[0]->shape_.size()};
  int *in_shape[1] = {in_tensors[0]->shape_.data()};
  int out_format;
  int out_datatype;
  int ret = ReduceInferShape(in_shape, dim_size, out_tensors[0]->shape_.data(), in_format, &out_format, in_datatype,
                             &out_datatype, param);
  if (ret != NNACL_OK) {
    LITE_ERROR_LOG("arithmetic infershape failed! ret: %d", ret);
    return RET_ERROR;
  }
  out_tensors[0]->data_type_ = in_tensors[0]->data_type_;
  out_tensors[0]->format_ = in_tensors[0]->format_;
  return RET_OK;
}

int DoReduce(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
             mindspore::lite::Allocator *allocator) {
  if (in_tensors.size() != 1 || in_tensors[0]->data_ == NULL) {
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

  ReduceParameter *params = reinterpret_cast<ReduceParameter *>(node->primitive_);
  Reducer reducer = NULL;
  if (params->mode_ == ReduceMode::ReduceMode_ReduceSum) {
    reducer = ReduceSum;
  } else if (params->mode_ == ReduceMode::ReduceMode_ReduceMean) {
    reducer = ReduceMean;
  }

  int buf_num = params->num_axes_ - 1;
  float *data_buffers[buf_num];
  int status = MallocTmpBuffer(data_buffers, in_tensors[0]->shape_, params->axes_, params->num_axes_, allocator);
  if (status != RET_OK) {
    FreeTmpBuffer(data_buffers, buf_num, allocator);
    return status;
  }

  status = RunReduce(reducer, data_buffers, reinterpret_cast<float *>(in_tensors[0]->data_),
                     reinterpret_cast<float *>(out_tensors[0]->data_), params, in_tensors[0]->shape_);

  FreeTmpBuffer(data_buffers, buf_num, allocator);

  if (status != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}
