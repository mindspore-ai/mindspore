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

#include "plugin/device/ascend/kernel/internal/activation.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalActivation::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  SetComputeType(param_ptr);
  return param_ptr;
}

void InternalActivation::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  outputsIdxMap_[kIndex0] = kIndex0;
}

class InternalSwish : public InternalActivation {
 public:
  InternalSwish() : InternalActivation("Swish") {}
  ~InternalSwish() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Swish;
    internal::ActivationParam op_param;
    op_param.activationType = internal::ActivationParam::ACTIVATION_SWISH;
    param_ptr->specificParam = op_param;
  }
};

class InternalSwiGLU : public InternalActivation {
 public:
  InternalSwiGLU() : InternalActivation("SwiGLU") {}
  ~InternalSwiGLU() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::SwiGLU;
    internal::ActivationParam op_param;
    op_param.activationType = internal::ActivationParam::ACTIVATION_SWIGLU_FORWARD;
    param_ptr->specificParam = op_param;
  }
};

MS_INTERNAL_KERNEL_FACTORY_REG(SiLU, InternalSwish);
}  // namespace kernel
}  // namespace mindspore
