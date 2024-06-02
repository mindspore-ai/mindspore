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
#include "plugin/device/ascend/kernel/opapi/aclnn/addmm_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void AddmmAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  auto beta_dtype_id = inputs[kIndex3]->dtype_id();
  switch (beta_dtype_id) {
    case kNumberTypeBool: {
      auto beta_value = inputs[kIndex3]->GetValueWithCheck<bool>();
      MAKE_SCALAR(beta_value, inputs[0]->dtype_id(), beta_);
      break;
    }
    case kNumberTypeFloat32: {
      auto beta_value = inputs[kIndex3]->GetValueWithCheck<float>();
      MAKE_SCALAR(beta_value, inputs[0]->dtype_id(), beta_);
      break;
    }
    case kNumberTypeFloat64: {
      auto beta_value = inputs[kIndex3]->GetValueWithCheck<double>();
      MAKE_SCALAR(beta_value, inputs[0]->dtype_id(), beta_);
      break;
    }
    case kNumberTypeInt64: {
      auto beta_value = inputs[kIndex3]->GetValueWithCheck<int64_t>();
      MAKE_SCALAR(beta_value, inputs[0]->dtype_id(), beta_);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Addmm beta only support bool, float32, float64 and int64, but got "
                        << TypeIdToString(beta_dtype_id);
  }
  auto alpha_dtype_id = inputs[kIndex4]->dtype_id();
  switch (alpha_dtype_id) {
    case kNumberTypeBool: {
      auto alpha_value = inputs[kIndex4]->GetValueWithCheck<bool>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    case kNumberTypeFloat32: {
      auto alpha_value = inputs[kIndex4]->GetValueWithCheck<float>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    case kNumberTypeFloat64: {
      auto alpha_value = inputs[kIndex4]->GetValueWithCheck<double>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    case kNumberTypeInt64: {
      auto alpha_value = inputs[kIndex4]->GetValueWithCheck<int64_t>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Addmm alpha only support bool, float32, float64 and int64, but got "
                        << TypeIdToString(alpha_dtype_id);
  }
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], beta_, alpha_, outputs[kIndex0],
                        OpApiUtil::GetCubeMathType());
}

bool AddmmAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], beta_,
                                      alpha_, outputs[kIndex0], OpApiUtil::GetCubeMathType()));
  RunOp(stream_ptr, workspace);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(Addmm, AddmmAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
