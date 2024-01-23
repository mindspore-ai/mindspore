/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/internal/gelu.h"

#include <memory>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalGelu::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::ActivationParam op_param;
  op_param.activationType = internal::ActivationParam::ACTIVATION_GELU;
  param_ptr->specificParam = op_param;
  param_ptr->opId = internal::OpId::Gelu;
  return param_ptr;
}

void InternalGelu::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  outputsIdxMap_[0] = 0;
}

int InternalGelu::Build(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto param = CreateOpParam(inputs, outputs);
  impl_ = internal::CreateInternalKernelImpl(param);

  // abstract validation info from inputs
  internal::ValidateInfo info;
  info.input_num_ = inputsIdxMap_.size();
  info.output_num_ = outputsIdxMap_.size();
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); iter++) {
    if (inputs[iter->first]->dtype_id() == TypeId::kNumberTypeBFloat16) {
      info.input_dtype_.emplace_back(internal::TensorDType::TENSOR_DTYPE_FLOAT16);
    } else {
      info.input_dtype_.emplace_back(InternalKernelUtils::ToInternalDType(inputs[iter->first]->dtype_id()));
    }
    info.input_format_.emplace_back(InternalKernelUtils::ToInternalFormat(inputs[iter->first]->format()));
  }
  for (auto iter = outputsIdxMap_.begin(); iter != outputsIdxMap_.end(); iter++) {
    if (outputs[iter->first]->dtype_id() == TypeId::kNumberTypeBFloat16) {
      info.output_dtype_.emplace_back(internal::TensorDType::TENSOR_DTYPE_FLOAT16);
    } else {
      info.output_dtype_.emplace_back(InternalKernelUtils::ToInternalDType(outputs[iter->first]->dtype_id()));
    }
    info.output_format_.emplace_back(InternalKernelUtils::ToInternalFormat(outputs[iter->first]->format()));
  }
  impl_->Init(info);
  for (auto iter = inputsIdxMap_.begin(); iter != inputsIdxMap_.end(); iter++) {
    InternalKernelUtils::ToInternalTensor(inputs_[iter->second], inputs[iter->first]);
  }
  impl_->SetInputs(inputs_);

  return 0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(GeLU, InternalGelu);
}  // namespace kernel
}  // namespace mindspore
