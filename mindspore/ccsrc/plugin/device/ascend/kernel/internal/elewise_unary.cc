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

#include "plugin/device/ascend/kernel/internal/elewise_unary.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "param/elewise_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr ElewiseUnary::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto elewise_param = std::make_shared<internal::ElewiseBinaryParam>();
  elewise_param->dtype_ = InternalKernelUtils::ToInternalDType(inputs[kIndex0]->dtype_id());

  auto param_ptr = std::static_pointer_cast<internal::OpParam>(elewise_param);
  SetComputeType(param_ptr);
  return param_ptr;
}

void ElewiseUnary::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  outputsIdxMap_[kIndex0] = kIndex0;
}

class InternalLogicalNot : public ElewiseUnary {
 public:
  InternalLogicalNot() : ElewiseUnary("LogicalNot") {}
  ~InternalLogicalNot() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::LogicalNot;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_LOGICAL_NOT;
    param_ptr->specificParam = op_param;
  }
};

MS_INTERNAL_KERNEL_FACTORY_REG(LogicalNot, InternalLogicalNot);
}  // namespace kernel
}  // namespace mindspore
