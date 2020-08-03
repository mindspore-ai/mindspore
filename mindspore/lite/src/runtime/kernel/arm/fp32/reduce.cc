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

#include "src/runtime/kernel/arm/fp32/reduce.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/opclib/fp32/reduce.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Mean;
using mindspore::schema::PrimitiveType_Reduce;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

int ReduceCPUKernel::CheckInputsOutputs() {
  if (inputs_.size() != kInputNum) {
    MS_LOG(ERROR) << "Reduce inputs size should be " << kInputNum << " but got " << inputs_.size();
    return RET_ERROR;
  }
  if (outputs_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Reduce outputs size should be " << kOutputNum << " but got " << outputs_.size();
    return RET_ERROR;
  }
  auto input = inputs_.at(0);
  if (input == nullptr) {
    MS_LOG(ERROR) << "Reduce input is nullptr";
    return RET_NULL_PTR;
  }
  auto output = outputs_.at(0);
  if (output == nullptr) {
    MS_LOG(ERROR) << "Reduce output is nullptr";
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int ReduceCPUKernel::CheckParameters() {
  size_t input_rank = inputs_.at(0)->shape().size();
  if (static_cast<size_t>(num_axes_) > input_rank) {
    MS_LOG(ERROR) << "Reduce num of reduce axes " << num_axes_ << " larger than input rank " << input_rank;
    return RET_ERROR;
  }
  for (auto i = 0; i < num_axes_; i++) {
    if (static_cast<size_t>(axes_[i]) < -input_rank || static_cast<size_t>(axes_[i]) >= input_rank) {
      MS_LOG(ERROR) << "Reduce got invalid axis " << axes_[i] << ", axis should be in [" << -input_rank << ", "
                    << input_rank - 1 << "].";
      return RET_ERROR;
    }
    if (axes_[i] < 0) {
      axes_[i] += input_rank;
    }
  }

  if (num_axes_ == 0) {
    for (int i = 0; i < input_rank; i++) {
      axes_[i] = i;
    }
  }

  return RET_OK;
}

int ReduceCPUKernel::Init() {
  auto ret = CheckInputsOutputs();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    return ret;
  }

  switch (mode_) {
    case static_cast<int>(ReduceMode_ReduceSum): {
      reducer_ = ReduceSum;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceMean): {
      reducer_ = ReduceMean;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceMax): {
      reducer_ = ReduceMax;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceMin): {
      reducer_ = ReduceMin;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceProd): {
      reducer_ = ReduceProd;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceSumSquare): {
      reducer_ = ReduceSumSquare;
      break;
    }
    default:
      MS_LOG(ERROR) << "Reduce unsupported reduce mode: " << mode_;
      return RET_ERROR;
  }
  return RET_OK;
}

int ReduceCPUKernel::CallReduceUnit(int task_id) {
  auto ret = reducer_(outer_size_, inner_size_, axis_size_, src_data_, tmp_shape_.data(), dst_data_, task_id,
                      context_->threadNum);
  return ret;
}

int ReduceImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto reduce = reinterpret_cast<ReduceCPUKernel *>(cdata);
  auto error_code = reduce->CallReduceUnit(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceCPUKernel::Run() {
  tmp_shape_ = inputs_.at(0)->shape();
  src_data_ = static_cast<float *>(inputs_.at(0)->Data());
  for (int i = 0; i < tmp_shape_.size(); ++i) {
    dst_data_ = data_buffers_[i];
    int axis = axes_[i];
    outer_size_ = 1;
    for (int j = 0; j < axis; j++) {
      outer_size_ *= tmp_shape_[j];
    }
    inner_size_ = 1;
    for (int k = axis + 1; k < static_cast<int>(tmp_shape_.size()); k++) {
      inner_size_ *= tmp_shape_[k];
    }
    axis_size_ = tmp_shape_[axis];
    auto error_code = LiteBackendParallelLaunch(ReduceImpl, this, context_->threadNum);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
    tmp_shape_[axis] = 1;
    src_data_ = dst_data_;
  }

  int last_reduce_axis = axes_[num_axes_ - 1];
  outer_size_ = 1;
  for (int i = 0; i < last_reduce_axis; i++) {
    outer_size_ *= tmp_shape_[i];
  }
  inner_size_ = 1;
  for (int i = last_reduce_axis + 1; i < static_cast<int>(tmp_shape_.size()); i++) {
    inner_size_ *= tmp_shape_[i];
  }
  axis_size_ = tmp_shape_[last_reduce_axis];
  dst_data_ = reinterpret_cast<float *>(outputs_.at(0)->Data());
  auto error_code = LiteBackendParallelLaunch(ReduceImpl, this, context_->threadNum);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

int ReduceCPUKernel::MallocTmpBuffer() {
  auto input_shape = inputs_.at(0)->shape();
  for (auto i = 0; i < num_axes_ - 1; i++) {
    int axis = axes_[i];
    size_t size = 1;
    for (auto j = 0; j < input_shape.size(); j++) {
      if (static_cast<size_t>(axis) != j) {
        size *= input_shape[j];
      }
    }
    float *buffer = reinterpret_cast<float *>(malloc(size * sizeof(float)));
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed.";
      return RET_ERROR;
    }
    data_buffers_.emplace_back(buffer);
    input_shape[axis] = 1;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuReduceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Reduce);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Reduce opParameter nullptr";
    return nullptr;
  }
  if (desc.type != schema::PrimitiveType_Reduce) {
    MS_LOG(ERROR) << "Reduce op desc.type should be PrimitiveType_Reduce, got " << desc.type;
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) ReduceCPUKernel(reinterpret_cast<ReduceParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Reduce new ReduceCPUKernel failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuMeanFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Mean);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Reduce opParameter nullptr";
    return nullptr;
  }
  if (desc.type != schema::PrimitiveType_Mean) {
    MS_LOG(ERROR) << "Reduce op desc.type should be PrimitiveType_Mean, got " << desc.type;
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) ReduceCPUKernel(reinterpret_cast<ReduceParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Reduce new ReduceCPUKernel failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Reduce, CpuReduceFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mean, CpuMeanFp32KernelCreator)
}  // namespace mindspore::kernel
