/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/base/scatter_nd_binary.h"
#include <cstring>
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ScatterNDBinaryCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ScatterNDBinaryCPUKernel::ReSize() {
  auto input = in_tensors_.at(kScatterUpdateInputIndex);
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  auto update = in_tensors_.at(kScatterUpdateIndex);
  auto input_shape = input->shape();
  auto indices_shape = indices->shape();
  auto update_shape = update->shape();
  const int input_rank = static_cast<int>(input_shape.size());
  const int indices_rank = static_cast<int>(indices_shape.size());
  const int update_rank = static_cast<int>(update_shape.size());
  const int indices_unit_rank = indices->shape().back();

  // check indices shape
  MS_CHECK_TRUE_MSG(indices_rank >= DIMENSION_2D, RET_ERROR, "The rank of indices must be greater equal than 2.");
  MS_CHECK_TRUE_MSG(indices_unit_rank <= input_rank, RET_ERROR,
                    "The value of indices' last dimension must be less equal than the input rank.");
  MS_CHECK_TRUE_MSG(update_rank == indices_rank - 1 + input_rank - indices_unit_rank, RET_ERROR,
                    "The rank of update is illegal.");
  // check consistency of the shape indices and shape
  for (int i = 0; i < update_rank; i++) {
    if (i < indices_rank - 1) {
      MS_CHECK_TRUE_MSG(update_shape[i] == indices_shape[i], RET_ERROR, "the shape of update tensor is illegal.");
    } else {
      MS_CHECK_TRUE_MSG(update_shape[i] == input_shape[indices_shape[indices_rank - 1] + i - indices_rank + 1],
                        RET_ERROR, "the shape of update tensor is illegal.");
    }
  }

  // calculate unit_size
  param_->unit_size = 1;
  for (int i = indices_rank - 1; i < update_rank; i++) {
    param_->unit_size *= update_shape.at(i);
  }

  // calculate offsets
  int out_stride = 1;
  std::vector<int> out_strides;
  out_strides.push_back(1);
  for (int i = indices_unit_rank - C2NUM; i >= 0; i--) {
    out_stride *= input_shape[i + 1];
    out_strides.push_back(out_stride);
  }
  std::reverse(out_strides.begin(), out_strides.end());

  param_->num_unit = 1;
  for (int i = indices_rank - C2NUM; i >= 0; i--) {
    param_->num_unit *= update_shape.at(i);
  }

  auto indices_ptr = indices->data();
  if (indices_ptr == nullptr) {
    return RET_OK;
  }
  output_unit_offsets_.clear();
  if (indices->data_type() == kNumberTypeInt32) {
    auto indices_data = reinterpret_cast<int *>(indices_ptr);
    for (int i = 0; i < param_->num_unit; i++) {
      int tmp_stride = 0;
      for (int j = 0; j < indices_unit_rank; j++) {
        tmp_stride += indices_data[i * indices_unit_rank + j] * out_strides.at(j) * param_->unit_size;
      }
      output_unit_offsets_.push_back(tmp_stride);
    }
  } else if (indices->data_type() == kNumberTypeInt64) {
    auto indices_data = reinterpret_cast<int64_t *>(indices_ptr);
    for (int i = 0; i < param_->num_unit; i++) {
      int tmp_stride = 0;
      for (int j = 0; j < indices_unit_rank; j++) {
        tmp_stride += indices_data[i * indices_unit_rank + j] * out_strides.at(j) * param_->unit_size;
      }
      output_unit_offsets_.push_back(tmp_stride);
    }
  } else {
    MS_LOG(ERROR) << "ScatterNDBinary only support int32 and int64 indices tensor, but got " << indices->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
