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
#include "src/runtime/kernel/arm/base/fullconnection_base.h"
#include "src/runtime/kernel/arm/int8/fullconnection_int8.h"
#include "src/runtime/kernel/arm/fp32/fullconnection.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int FullconnectionBaseCPUKernel::Init() {
  fc_param_->op_parameter_.thread_num_ = thread_count_;
  return RET_OK;
}

kernel::LiteKernel *CpuFullConnectionInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Concat);
  auto kernel = new (std::nothrow) FullconnectionInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (!kernel) {
    MS_LOG(ERROR) << "kernel is nullptr.";
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
int RestoreFullconnectWeight(lite::tensor::Tensor *input_tensor) {
  MS_ASSERT(input_tensor != nullptr);
  if (input_tensor->data_type() != kNumberTypeUInt8) {
    MS_LOG(ERROR) << "full connect input type error" << input_tensor->data_type();
    return RET_ERROR;
  }
  if (input_tensor->GetQuantParams().empty()) {
    MS_LOG(ERROR) << "no quant param";
    return RET_ERROR;
  }
  const auto* quant_data = static_cast<const uint8_t*>(input_tensor->Data());
  auto* dequant_data = static_cast<float *>(malloc(input_tensor->DataSize() * sizeof(float)));
  if (dequant_data == nullptr) {
    MS_LOG(ERROR) << "malloc faile";
    return RET_ERROR;
  }

  if (input_tensor->GetQuantParams().size() != kPerTensor) {
    size_t channels = static_cast<size_t>(input_tensor->Batch());
    if (input_tensor->GetQuantParams().size() != channels) {
      MS_LOG(ERROR) << "Quant param not equal channel num " << input_tensor->GetQuantParams().size() << channels;
      return RET_ERROR;
    }
    size_t per_channel_size = input_tensor->DataSize() / channels;
    auto quant_param = input_tensor->GetQuantParams();
    for (size_t i = 0; i < channels; i++) {
      auto param = quant_param.at(i);
      auto scale = param.scale;
      auto zero_point = param.zeroPoint;
      for (size_t j = 0; j < per_channel_size; j++) {
        dequant_data[per_channel_size * i + j] = static_cast<float>(
          (quant_data[per_channel_size * i + j] - zero_point) * scale);
      }
    }
  } else {
    auto quant_param = input_tensor->GetQuantParams();
    auto param = quant_param.front();
    auto scale = param.scale;
    auto zero_point = param.zeroPoint;
    for (int64_t j = 0; j < input_tensor->DataSize(); j++) {
      dequant_data[j] = static_cast<float>((quant_data[j] - zero_point) * scale);
    }
  }
  input_tensor->SetData(dequant_data);
  return RET_OK;
}
kernel::LiteKernel *CpuFullConnectionFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Concat);
  auto *weight_tensor = inputs.at(kWeightIndex);
  auto *restore_data = weight_tensor->Data();
  if (primitive->GetQuantType() == schema::QuantType_WeightQuant) {
    RestoreFullconnectWeight(inputs.at(kWeightIndex));
  }
  auto kernel = new (std::nothrow) FullconnectionCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (!kernel) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  if (primitive->GetQuantType() == schema::QuantType_WeightQuant) {
    weight_tensor->FreeData();
    weight_tensor->SetData(restore_data);
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_FullConnection, CpuFullConnectionInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FullConnection, CpuFullConnectionFp32KernelCreator)
}  // namespace mindspore::kernel
