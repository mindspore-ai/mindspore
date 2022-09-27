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

#include "src/litert/kernel/cpu/fp32/tensor_scatter_add.h"
#include <cstring>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorScatterAdd;

namespace mindspore::kernel {
namespace {
constexpr int kScatterUpdateInputIndex = 0;
constexpr int kScatterIndicesIndex = 1;
constexpr int kScatterUpdateIndex = 2;

int TensorScatterAddRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = static_cast<TensorScatterAddCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->TensorScatterAdd(task_id);
}
}  // namespace

int TensorScatterAddCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TensorScatterAddCPUKernel::ReSize() {
  auto input = in_tensors_.at(kScatterUpdateInputIndex);
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  auto update = in_tensors_.at(kScatterUpdateIndex);
  auto input_shape = input->shape();
  auto indices_shape = indices->shape();
  auto update_shape = update->shape();
  int input_rank = static_cast<int>(input->shape().size());
  int indices_rank = static_cast<int>(indices->shape().size());
  int update_rank = static_cast<int>(update->shape().size());
  int indice_unit_rank = indices->shape().back();

  // check indices shape
  MS_CHECK_TRUE_MSG(indices_rank >= DIMENSION_2D, RET_ERROR, "The rank of indices must be greater equal than 2.");
  MS_CHECK_TRUE_MSG(indice_unit_rank <= input_rank, RET_ERROR,
                    "The value of indices' last dimension must be less equal than the input rank.");
  MS_CHECK_TRUE_MSG(update_rank == indices_rank - 1 + input_rank - indice_unit_rank, RET_ERROR,
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
  for (int i = indices_shape.size() - 1; i < update_rank; i++) {
    param_->unit_size *= update_shape.at(i);
  }

  // calculate offsets
  int out_stride = 1;
  std::vector<int> out_strides;
  out_strides.push_back(1);
  for (int i = indice_unit_rank - C2NUM; i >= 0; i--) {
    out_stride *= input_shape[i + 1];
    out_strides.push_back(out_stride);
  }
  std::reverse(out_strides.begin(), out_strides.end());

  param_->num_unit = 1;
  param_->num_unit *= update_shape.at(indices_shape.size() - C2NUM);
  for (int i = indices_shape.size() - C3NUM; i >= 0; i--) {
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
      for (int j = 0; j < indice_unit_rank; j++) {
        tmp_stride += indices_data[i * indice_unit_rank + j] * out_strides.at(j) * param_->unit_size;
      }
      output_unit_offsets_.push_back(tmp_stride);
    }
  } else if (indices->data_type() == kNumberTypeInt64) {
    auto indices_data = reinterpret_cast<int64_t *>(indices_ptr);
    for (int i = 0; i < param_->num_unit; i++) {
      int tmp_stride = 0;
      for (int j = 0; j < indice_unit_rank; j++) {
        tmp_stride += indices_data[i * indice_unit_rank + j] * out_strides.at(j) * param_->unit_size;
      }
      output_unit_offsets_.push_back(tmp_stride);
    }
  } else {
    MS_LOG(ERROR) << "TensorScatterAdd only support int32 and int64 indices tensor, but got " << indices->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorScatterAddCPUKernel::TensorScatterAdd(int task_id) {
  auto data_type = in_tensors_[kScatterUpdateInputIndex]->data_type();
  if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "TensorScatterAdd only support int32 and float32 input tensor, but got " << data_type;
    return RET_ERROR;
  }
  int type = data_type == kNumberTypeFloat32 ? 0 : 1;
  auto ret = ScatterNDAdd(in_tensors_[kScatterUpdateIndex]->data(), out_tensors_[kOutputIndex]->data(),
                          output_unit_offsets_.data(), param_, type, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoScatterND failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorScatterAddCPUKernel::Run() {
  auto in_tensor = in_tensors().front();
  auto out_tensor = out_tensors().front();
  memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  if (!indices->IsConst() && ReSize() != RET_OK) {
    MS_LOG(ERROR) << "TensorScatterAdd resize failed.";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(ms_context_, TensorScatterAddRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorScatterAdd error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorScatterAdd, LiteKernelCreator<TensorScatterAddCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorScatterAdd, LiteKernelCreator<TensorScatterAddCPUKernel>)
}  // namespace mindspore::kernel
