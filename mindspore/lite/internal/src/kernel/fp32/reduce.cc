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
#include <vector>
#include "internal/include/model.h"
#include "internal/include/lite_utils.h"
#include "src/runtime/allocator.h"
#include "internal/src/lite_log.h"
#include "internal/include/errorcode.h"
#include "nnacl/reduce_parameter.h"
#include "nnacl/fp32/reduce.h"
#include "schema/ops_generated.h"

typedef int (*Reducer)(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                       float *dst_data, const int tid, const int thread_num);

int MallocTmpBuffer(std::vector<float *> *data_buffers, const ShapeVector &shape, const int *axes, const int num_axes,
                    mindspore::lite::Allocator *allocator) {
  for (int i = 0; i < data_buffers->size(); ++i) {
    if (data_buffers->at(i) != NULL) {
      free(data_buffers->at(i));
      data_buffers->at(i) = NULL;
    }
  }
  data_buffers->clear();

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
    if (buffer == NULL) {
      LITE_ERROR_LOG("Memory allocation failed!")
      return RET_ERROR;
    }
    data_buffers->emplace_back(buffer);
    input_shape[axis] = 1;
  }
  return RET_OK;
}

int FreeTmpBuffer(std::vector<float *> *data_buffers, mindspore::lite::Allocator *allocator) {
  for (int i = 0; i < data_buffers->size(); ++i) {
    allocator->Free(data_buffers->at(i));
  }
  data_buffers->clear();
  return RET_OK;
}

int RunReduce(Reducer reducer, std::vector<float *> data_buffers, float *in_data, float *out_data, Int32Vector axes,
              ShapeVector shape) {
  int rank = shape.size();
  float *dst_data = NULL;
  float *src_data = in_data;
  ShapeVector tmp_shape = shape;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i != axes.size() - 1) {
      dst_data = data_buffers[i];
    } else {
      dst_data = out_data;
    }
    int axis = axes[i];
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
      LITE_ERROR_LOG("Reduce run error!")
      return RET_ERROR;
    }
    tmp_shape[axis] = 1;
    src_data = dst_data;
  }
  return RET_OK;
}

int DoReduceInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param) {
  if (in_tensors.size() != 1 || in_tensors[0]->data_ == NULL) {
    LITE_ERROR_LOG("input tensors num not correct or input data is NULL!")
    return RET_INPUT_TENSOR_ERROR;
  }
  if (out_tensors.size() != 1) {
    LITE_ERROR_LOG("output tensors num not correct!")
    return RET_ERROR;
  }

  ReduceParameter *reduceParameter = reinterpret_cast<ReduceParameter *>(param);
  bool keep_dims = reduceParameter->keep_dims_;
  int num_axes = reduceParameter->num_axes_;
  ShapeVector in_shape = in_tensors[0]->shape_;
  int rank = in_shape.size();
  Int32Vector out_shape;
  Int32Vector axes;
  int actual_axes_num = num_axes;
  for (int i = 0; i < num_axes; ++i) {
    if (reduceParameter->axes_[i] < -rank || reduceParameter->axes_[i] >= rank) {
      LITE_ERROR_LOG("reduce_sum got invalid axis!")
      return RET_ERROR;
    }
    if (reduceParameter->axes_[i] < 0) {
      axes.push_back(reduceParameter->axes_[i] + rank);
    } else {
      axes.push_back(reduceParameter->axes_[i]);
    }
  }
  if (reduceParameter->reduce_to_end_) {
    if (num_axes != 1) {
      LITE_ERROR_LOG("Reduce when reduce_to_end, num of axis should be 1!")
      return RET_ERROR;
    }
    int begin_axis = axes[0];
    num_axes = rank - begin_axis;
    for (auto i = begin_axis + 1; i < rank; ++i) {
      axes[actual_axes_num++] = i;
    }
  }

  if (num_axes == 0) {
    axes.resize(rank);
    for (size_t i = 0; i < rank; i++) {
      axes[i] = i;
      if (keep_dims) {
        out_shape.push_back(1);
      }
    }
    reduceParameter->num_axes_ = axes.size();
    for (int i = 0; i < axes.size(); ++i) {
      reduceParameter->axes_[i] = axes[i];
    }
    out_tensors[0]->shape_ = out_shape;
    out_tensors[0]->data_type_ = in_tensors[0]->data_type_;
    out_tensors[0]->format_ = in_tensors[0]->format_;
    return RET_OK;
  }
  // reduce on selected axes
  for (size_t i = 0; i < rank; i++) {
    bool reduce_axis = false;
    for (size_t idx = 0; idx < num_axes; ++idx) {
      if (axes[idx] == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }
  reduceParameter->num_axes_ = axes.size();
  for (int i = 0; i < axes.size(); ++i) {
    reduceParameter->axes_[i] = axes[i];
  }
  out_tensors[0]->shape_ = out_shape;
  out_tensors[0]->data_type_ = in_tensors[0]->data_type_;
  out_tensors[0]->format_ = in_tensors[0]->format_;
  return RET_OK;
}

int DoReduce(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
             mindspore::lite::Allocator *allocator) {
  if (in_tensors.size() != 1 || in_tensors[0]->data_ == NULL) {
    LITE_ERROR_LOG("input tensors num not correct or input data is NULL!")
    return RET_INPUT_TENSOR_ERROR;
  }
  if (out_tensors.size() != 1 || out_tensors[0]->data_ == NULL) {
    LITE_ERROR_LOG("output tensors num not correct or output data is NULL!")
    return RET_ERROR;
  }
  if (allocator == NULL) {
    LITE_ERROR_LOG("allocator is NULL!")
    return RET_ERROR;
  }

  ReduceParameter *params = reinterpret_cast<ReduceParameter *>(node->primitive_);
  Reducer reducer = NULL;
  if (params->mode_ == mindspore::schema::ReduceMode::ReduceMode_ReduceSum) {
    reducer = ReduceSum;
  } else if (params->mode_ == mindspore::schema::ReduceMode::ReduceMode_ReduceMean) {
    reducer = ReduceMean;
  }

  std::vector<float *> data_buffers;
  int status = MallocTmpBuffer(&data_buffers, in_tensors[0]->shape_, params->axes_, params->num_axes_, allocator);
  if (status != RET_OK) {
    FreeTmpBuffer(&data_buffers, allocator);
    return status;
  }

  Int32Vector axes;
  for (int i = 0; i < params->num_axes_; ++i) {
    axes.push_back(params->axes_[i]);
  }
  status = RunReduce(reducer, data_buffers, reinterpret_cast<float *>(in_tensors[0]->data_),
                     reinterpret_cast<float *>(out_tensors[0]->data_), axes, in_tensors[0]->shape_);
  if (status != RET_OK) {
    return status;
  }

  status = FreeTmpBuffer(&data_buffers, allocator);
  if (status != RET_OK) {
    return status;
  }
  return RET_OK;
}
