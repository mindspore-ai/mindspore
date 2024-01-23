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
#include "plugin/device/ascend/kernel/internal/elewise_binary.h"
#include <memory>

namespace mindspore {
namespace kernel {
internal::OpParamPtr ElewiseBinary::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  SetComputeType(param_ptr);
  return param_ptr;
}

void ElewiseBinary::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  outputsIdxMap_[0] = 0;
}

class InternalAdd : public ElewiseBinary {
 public:
  InternalAdd() : ElewiseBinary("Add") {}
  ~InternalAdd() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Add;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_ADD;
    param_ptr->specificParam = op_param;
  }
};

class InternalSub : public ElewiseBinary {
 public:
  InternalSub() : ElewiseBinary("Sub") {}
  ~InternalSub() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Sub;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_SUB;
    param_ptr->specificParam = op_param;
  }
};

MS_INTERNAL_KERNEL_FACTORY_REG(Add, InternalAdd);
MS_INTERNAL_KERNEL_FACTORY_REG(Sub, InternalSub);
}  // namespace kernel
}  // namespace mindspore
