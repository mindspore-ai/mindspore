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

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "param/add_param.h"
#include "param/sub_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr ElewiseBinary::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  SetComputeType(param_ptr);
  return param_ptr;
}

void ElewiseBinary::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  inputsIdxMap_[kIndex1] = kIndex1;
  outputsIdxMap_[kIndex0] = kIndex0;
}

uint64_t ElewiseBinary::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(kernel_name_, inputs[kIndex0]->GetShapeVector(),
                                                         inputs[kIndex0]->dtype_id(), inputs[kIndex1]->GetShapeVector(),
                                                         inputs[kIndex1]->dtype_id());
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
    param_ptr->input1_dtype_ = InternalKernelUtils::ToInternalDType(inputs[kIndex0]->dtype_id());
    param_ptr->input2_dtype_ = InternalKernelUtils::ToInternalDType(inputs[kIndex1]->dtype_id());
    param_ptr->input1_dims_ = internal::VecToSVec<int64_t>(inputs[kIndex0]->GetShapeVector());
    param_ptr->input2_dims_ = internal::VecToSVec<int64_t>(inputs[kIndex1]->GetShapeVector());
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_ADD;
    param_ptr->specificParam = op_param;
    return std::static_pointer_cast<internal::OpParam>(param_ptr);
  }
};

class InternalSub : public ElewiseBinary {
 public:
  InternalSub() : ElewiseBinary("Sub") {}
  ~InternalSub() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {}
  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) override {
    auto param_ptr = std::make_shared<internal::SubParam>();
    param_ptr->opId = internal::OpId::Sub;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_SUB;
    param_ptr->specificParam = op_param;

    param_ptr->input1_dtype_ = InternalKernelUtils::ToInternalDType(inputs[kIndex0]->dtype_id());
    param_ptr->input2_dtype_ = InternalKernelUtils::ToInternalDType(inputs[kIndex1]->dtype_id());
    param_ptr->input1_dims_ = internal::VecToSVec<int64_t>(inputs[kIndex0]->GetShapeVector());
    param_ptr->input2_dims_ = internal::VecToSVec<int64_t>(inputs[kIndex1]->GetShapeVector());
    
    return std::static_pointer_cast<internal::OpParam>(param_ptr);
  }
};

class InternalEqual : public ElewiseBinary {
 public:
  InternalEqual() : ElewiseBinary("Equal") {}
  ~InternalEqual() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Equal;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_EQUAL;
    param_ptr->specificParam = op_param;
  }
};

class InternalNotEqual : public ElewiseBinary {
 public:
  InternalNotEqual() : ElewiseBinary("NotEqual") {}
  ~InternalNotEqual() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::NotEqual;
    return;
  }
};
MS_INTERNAL_KERNEL_FACTORY_REG(NotEqual, InternalNotEqual);

class InternalLess : public ElewiseBinary {
 public:
  InternalLess() : ElewiseBinary("Less") {}
  ~InternalLess() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Less;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_LESS;
    param_ptr->specificParam = op_param;
  }
};

class InternalMul : public ElewiseBinary {
 public:
  InternalMul() : ElewiseBinary("Mul") {}
  ~InternalMul() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::Mul;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_MUL;
    param_ptr->specificParam = op_param;
  }
};

class InternalRealDiv : public ElewiseBinary {
 public:
  InternalRealDiv() : ElewiseBinary("RealDiv") {}
  ~InternalRealDiv() = default;

 protected:
  void SetComputeType(internal::OpParamPtr param_ptr) override {
    param_ptr->opId = internal::OpId::RealDiv;
    internal::ElewiseParam op_param;
    op_param.elewiseType = internal::ElewiseParam::ELEWISE_REALDIV;
    param_ptr->specificParam = op_param;
  }
};

MS_INTERNAL_KERNEL_FACTORY_REG(Add, InternalAdd);
MS_INTERNAL_KERNEL_FACTORY_REG(Sub, InternalSub);
MS_INTERNAL_KERNEL_FACTORY_REG(Equal, InternalEqual);
MS_INTERNAL_KERNEL_FACTORY_REG(Less, InternalLess);
MS_INTERNAL_KERNEL_FACTORY_REG(Mul, InternalMul);
MS_INTERNAL_KERNEL_FACTORY_REG(RealDiv, InternalRealDiv);
}  // namespace kernel
}  // namespace mindspore
