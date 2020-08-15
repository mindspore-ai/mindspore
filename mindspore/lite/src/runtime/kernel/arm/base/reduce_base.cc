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

#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/base/reduce_base.h"
#include "src/runtime/kernel/arm/fp32/reduce.h"
#include "src/runtime/kernel/arm/int8/reduce_int8.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Mean;
using mindspore::schema::PrimitiveType_Reduce;

namespace mindspore::kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

int ReduceBaseCPUKernel::CheckInputsOutputs() {
  if (in_tensors_.size() != kInputNum) {
    MS_LOG(ERROR) << "Reduce inputs size should be " << kInputNum << " but got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Reduce outputs size should be " << kOutputNum << " but got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto input = in_tensors_.at(0);
  if (input == nullptr) {
    MS_LOG(ERROR) << "Reduce input is nullptr";
    return RET_NULL_PTR;
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
    if (axes_[i] < -static_cast<int>(input_rank) || axes_[i] >= static_cast<int>(input_rank)) {
      MS_LOG(ERROR) << "Reduce got invalid axis " << axes_[i] << ", axis should be in ["
                    << -static_cast<int>(input_rank) << ", " << input_rank - 1 << "].";
      return RET_ERROR;
    }
    if (axes_[i] < 0) {
      axes_[i] += static_cast<int>(input_rank);
    }
  }

  if (num_axes_ == 0) {
    for (int i = 0; i < input_rank; i++) {
      axes_[i] = i;
    }
    num_axes_ = static_cast<int>(input_rank);
  }

  return RET_OK;
}

int ReduceBaseCPUKernel::Init() {
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (reduce_param == nullptr) {
    return RET_NULL_PTR;
  }
  num_axes_ = reduce_param->num_axes_;
  mode_ = reduce_param->mode_;
  memcpy(axes_, reduce_param->axes_, sizeof(reduce_param->axes_));

  auto ret = CheckInputsOutputs();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuReduceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
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
  auto *kernel = new (std::nothrow) ReduceCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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
                                             const kernel::KernelKey &desc, const lite::Primitive *primitive) {
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
  auto *kernel = new (std::nothrow) ReduceCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

kernel::LiteKernel *CpuReduceInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
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
  auto *kernel = new (std::nothrow) ReduceInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
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
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Reduce, CpuReduceInt8KernelCreator)
}  // namespace mindspore::kernel
