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

namespace mindspore {
namespace kernel {
internal::OpParamPtr ElewiseUnary::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  SetComputeType(param_ptr);
  return param_ptr;
}

void ElewiseUnary::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  outputsIdxMap_[0] = 0;
}

class InternalCast : public ElewiseUnary {
 public:
  InternalCast() : ElewiseUnary("Cast") {}
  ~InternalCast() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override { return; }

  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) override {
    internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
    param_ptr->opId = internal::OpId::Cast;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_CAST;
    if (inputs[0]->dtype_id() == TypeId::kNumberTypeFloat16 && outputs[0]->dtype_id() == TypeId::kNumberTypeBFloat16) {
      op_param.outTensorType = internal::TensorDType::TENSOR_DTYPE_UNDEFINED;
    } else {
      op_param.outTensorType = InternalKernelUtils::ToInternalDType(outputs[0]->dtype_id());
    }
    param_ptr->specificParam = op_param;
    return param_ptr;
  }
};

MS_INTERNAL_KERNEL_FACTORY_REG(Cast, InternalCast);
}  // namespace kernel
}  // namespace mindspore
