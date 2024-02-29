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
#include <memory>
#include "plugin/device/ascend/kernel/internal/elewise_binary.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "include/param/add_param.h"

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
  void SetComputeType(internal::OpParamPtr param_ptr) override {}
  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) override {
    auto param_ptr = std::make_shared<internal::AddParam>();
    param_ptr->opId = internal::OpId::Add;
    param_ptr->input1_dtype_ = InternalKernelUtils::ToInternalDType(inputs[0]->dtype_id());
    param_ptr->input2_dtype_ = InternalKernelUtils::ToInternalDType(inputs[1]->dtype_id());
    param_ptr->input1_dims_ = internal::VecToSVec<int64_t>(inputs[0]->GetShapeVector());
    param_ptr->input2_dims_ = internal::VecToSVec<int64_t>(inputs[1]->GetShapeVector());
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_ADD;
    param_ptr->specificParam = op_param;
    return std::static_pointer_cast<internal::OpParam>(param_ptr);
  }
  uint64_t GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                             const std::vector<KernelTensor *> &outputs) override {
    return TilingCacheMgr::GetInstance().GenTilingCacheKey(kernel_name_, inputs[0]->GetShapeVector(),
                                                           inputs[0]->dtype_id(), inputs[1]->GetShapeVector(),
                                                           inputs[1]->dtype_id());
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
