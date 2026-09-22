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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inplace_normal_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace inplace_normal {
float GetInputValueToFloat(const std::vector<KernelTensor *> &inputs, const size_t kIndex) {
  float value = 0;
  auto dtype_id = inputs[kIndex]->dtype_id();

  switch (dtype_id) {
    case kNumberTypeFloat32: {
      value = inputs[kIndex]->GetValueWithCheck<float>();
      break;
    }
    case kNumberTypeFloat64: {
      value = static_cast<float>(inputs[kIndex]->GetValueWithCheck<double>());
      break;
    }
    case kNumberTypeInt32: {
      value = static_cast<float>(inputs[kIndex]->GetValueWithCheck<int32_t>());
      break;
    }
    case kNumberTypeInt64: {
      value = static_cast<float>(inputs[kIndex]->GetValueWithCheck<int64_t>());
      break;
    }
    case kNumberTypeBool: {
      value = inputs[kIndex]->GetValueWithCheck<bool>() ? 1.0f : 0.0f;
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "mean and std parameters of InplaceNormal only support bool, int and float but got "
                        << dtype_id;
  }
  return value;
}

void InplaceNormalAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  mean_ = GetInputValueToFloat(inputs, kIndex1);
  std_ = GetInputValueToFloat(inputs, kIndex2);
  seed_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  offset_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);

  GetWorkspaceForResize(inputs[kIndex0], mean_, std_, seed_, offset_);
}

bool InplaceNormalAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  RunOp(stream_ptr, workspace, inputs[kIndex0], mean_, std_, seed_, offset_);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(InplaceNormal, InplaceNormalAscend);
}  // namespace inplace_normal
}  // namespace kernel
}  // namespace mindspore
