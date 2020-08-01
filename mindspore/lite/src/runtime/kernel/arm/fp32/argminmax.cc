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

#include "src/runtime/kernel/arm/fp32/argminmax.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/fp32/arg_min_max.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ArgMax;
using mindspore::schema::PrimitiveType_ArgMin;

namespace mindspore::kernel {
namespace {
constexpr int kInputNum = 1;
constexpr int kOutputNum = 1;
}  // namespace

int ArgMinMaxCPUKernel::Init() {
  switch (opParameter->type_) {
    case PrimitiveType_ArgMax:
      get_max_ = true;
      break;
    case PrimitiveType_ArgMin:
      get_max_ = false;
      break;
    default:
      MS_LOG(ERROR) << "Unexpected type " << opParameter->type_;
      return RET_ERROR;
  }
  auto dims_size = inputs_.at(0)->shape().size();
  axis_ = reinterpret_cast<ArgMinMaxParameter *>(opParameter)->axis_;
  axis_ = axis_ < 0 ? axis_ + dims_size : axis_;
  return RET_OK;
}

int ArgMinMaxCPUKernel::Run() {
  auto input = inputs_.at(0);

  auto input_data = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto output_data = reinterpret_cast<float *>(outputs_.at(0)->Data());

  auto shape = input->shape().data();
  int dims_number = input->shape().size();
  bool out_value = reinterpret_cast<ArgMinMaxParameter *>(opParameter)->out_value_;
  if (get_max_) {
    ArgMax(input_data, shape, dims_number, axis_, out_value, output_data);
  } else {
    ArgMin(input_data, shape, dims_number, axis_, out_value, output_data);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArgMinMaxFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ArgMinMaxCPUKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMax, CpuArgMinMaxFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMin, CpuArgMinMaxFp32KernelCreator)
}  // namespace mindspore::kernel
