/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/base/reduce_base.h"
#include <set>
#include "src/litert/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/litert/kernel/cpu/fp32/reduce_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

int ReduceBaseCPUKernel::CheckInputsOutputs() {
  if (in_tensors_.size() < kInputNum) {
    MS_LOG(ERROR) << "Reduce inputs size should be at least " << kInputNum << " but got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Reduce outputs size should be " << kOutputNum << " but got " << out_tensors_.size();
    return RET_ERROR;
  }
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    if (in_tensors_.at(i) == nullptr) {
      MS_LOG(ERROR) << "Reduce input is nullptr";
      return RET_NULL_PTR;
    }
  }
  auto output = out_tensors_.at(0);
  if (output == nullptr) {
    MS_LOG(ERROR) << "Reduce output is nullptr";
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int ReduceBaseCPUKernel::CheckParameters() {
  size_t input_rank = in_tensors_.at(0)->shape().size();
  if (static_cast<size_t>(num_axes_) > input_rank) {
    MS_LOG(ERROR) << "Reduce op invalid num of reduce axes " << num_axes_ << " larger than input rank " << input_rank;
    return RET_ERROR;
  }

  for (auto i = 0; i < num_axes_; i++) {
    if (axes_[i] < -(static_cast<int>(input_rank)) || axes_[i] >= static_cast<int>(input_rank)) {
      MS_LOG(ERROR) << "Reduce got invalid axis " << axes_[i] << ", axis should be in ["
                    << -(static_cast<int>(input_rank)) << ", " << input_rank - 1 << "].";
      return RET_ERROR;
    }
    if (axes_[i] < 0) {
      axes_[i] += static_cast<int>(input_rank);
    }
  }

  if (reduce_to_end_) {
    // actual num of axes to reduce
    num_axes_ = static_cast<int>(input_rank) - axes_[0];
    for (auto i = 1; i < num_axes_; ++i) {
      axes_[i] = axes_[0] + i;
    }
  }

  if (num_axes_ == 0) {
    for (size_t i = 0; i < input_rank; i++) {
      axes_[i] = static_cast<int>(i);
    }
    num_axes_ = static_cast<int>(input_rank);
  }
  return RET_OK;
}

int ReduceBaseCPUKernel::Prepare() {
  auto ret = CheckInputsOutputs();
  if (ret != RET_OK) {
    return ret;
  }

  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (reduce_param == nullptr) {
    return RET_NULL_PTR;
  }
  MS_CHECK_FALSE_MSG(op_parameter_->thread_num_ == 0, RET_ERROR, "thread_num_ should not be 0");
  if (in_tensors_.size() > 1) {
    auto axes_tensor = in_tensors_.at(1);
    MS_CHECK_TRUE_MSG(axes_tensor != nullptr, RET_ERROR, "axes-tensor is a nullptr.");
    MS_CHECK_FALSE_MSG((axes_tensor->data_type() != kNumberTypeInt && axes_tensor->data_type() != kNumberTypeInt32),
                       RET_ERROR, "The data type of axes tensor should be int32");
    num_axes_ = static_cast<int>(axes_tensor->ElementsNum());
    if (axes_tensor->data() != nullptr && (num_axes_ <= 0 || num_axes_ > MAX_SHAPE_SIZE)) {
      MS_LOG(ERROR) << "input axes invalid.";
      return RET_ERROR;
    }
    if (axes_tensor->data() == nullptr) {
      num_axes_ = static_cast<int>(in_tensors_.at(0)->shape().size());
      for (auto i = 0; i < num_axes_; i++) {
        axes_[i] = i;
      }
    } else {
      MS_CHECK_FALSE(axes_tensor->Size() == 0, RET_ERROR);
      (void)memcpy(axes_, axes_tensor->data(), axes_tensor->Size());
    }
  } else {
    num_axes_ = reduce_param->num_axes_;
    (void)memcpy(axes_, reduce_param->axes_, sizeof(reduce_param->axes_));
  }

  mode_ = reduce_param->mode_;
  reduce_to_end_ = reduce_param->reduce_to_end_;
  return RET_OK;
}

void ReduceBaseCPUKernel::CalculateInnerOuterSize() {
  outer_sizes_.clear();
  inner_sizes_.clear();
  axis_sizes_.clear();
  auto tmp_shape = in_tensors_.at(0)->shape();
  for (auto i = 0; i < num_axes_; ++i) {
    int axis = axes_[i];
    auto outer_size = 1;
    for (int j = 0; j < axis; j++) {
      outer_size *= tmp_shape.at(j);
    }
    outer_sizes_.emplace_back(outer_size);
    auto inner_size = 1;
    for (int k = axis + 1; k < static_cast<int>(tmp_shape.size()); k++) {
      inner_size *= tmp_shape.at(k);
    }
    inner_sizes_.emplace_back(inner_size);
    axis_sizes_.emplace_back(tmp_shape.at(axis));
    tmp_shape.at(axis) = 1;
  }
}

void ReduceBaseCPUKernel::CalculateTmpBufferSize() {
  buffer_sizes_.clear();
  auto input_shape = in_tensors_.at(0)->shape();
  // calculate size of buffer to malloc for each reducing axis
  for (auto i = 0; i < num_axes_ - 1; i++) {
    int axis = axes_[i];
    size_t size = 1;
    for (size_t j = 0; j < input_shape.size(); j++) {
      if (axis != static_cast<int>(j)) {
        size *= static_cast<size_t>(input_shape.at(j));
      }
    }
    buffer_sizes_.emplace_back(size);
    input_shape.at(axis) = 1;
  }
}

int ReduceBaseCPUKernel::ReSize() {
  auto ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  DecideIfOnlyCopy();
  CalculateTmpBufferSize();
  CalculateInnerOuterSize();
  return RET_OK;
}

void ReduceBaseCPUKernel::DecideIfOnlyCopy() {
  auto in_shape = in_tensors_[FIRST_INPUT]->shape();
  std::set<int> can_not_copy = {schema::ReduceMode_ReduceSumSquare, schema::ReduceMode_ReduceASum,
                                schema::ReduceMode_ReduceAll, schema::ReduceMode_ReduceL2};
  if (can_not_copy.find(mode_) != can_not_copy.end()) {
    only_copy_ = false;
    return;
  }
  if (std::all_of(axes_, axes_ + num_axes_, [&in_shape](int axis) { return in_shape[axis] == 1; })) {
    only_copy_ = true;
  } else {
    only_copy_ = false;
  }
}

int ReduceBaseCPUKernel::CopyInputToOutput() {
  auto in_tensor = in_tensors().front();
  CHECK_NULL_RETURN(in_tensor);
  auto out_tensor = out_tensors().front();
  CHECK_NULL_RETURN(out_tensor);
  if (in_tensor->allocator() == nullptr || in_tensor->allocator() != out_tensor->allocator() ||
      in_tensor->allocator() != ms_context_->allocator || op_parameter_->is_train_session_) {
    CHECK_NULL_RETURN(out_tensor->data());
    CHECK_NULL_RETURN(in_tensor->data());
    MS_CHECK_FALSE(in_tensor->Size() == 0, RET_ERROR);
    if (in_tensor->data() != out_tensor->data()) {
      memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
    }
    return RET_OK;
  }

  out_tensor->FreeData();
  out_tensor->ResetRefCount();
  out_tensor->set_data(in_tensor->data());
  if (in_tensor->IsConst()) {
    out_tensor->set_own_data(false);
  } else {
    out_tensor->set_own_data(in_tensor->own_data());
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
