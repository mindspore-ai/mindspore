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
#include "plugin/device/ascend/kernel/opapi/aclnn/sub_ext_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void SubExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto alpha_dtype_id = inputs[kIndex2]->dtype_id();
  switch (alpha_dtype_id) {
    case kNumberTypeBool: {
      auto alpha_value = inputs[kIndex2]->GetValueWithCheck<bool>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    case kNumberTypeFloat32: {
      auto alpha_value = inputs[kIndex2]->GetValueWithCheck<float>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    case kNumberTypeFloat64: {
      auto alpha_value = inputs[kIndex2]->GetValueWithCheck<double>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    case kNumberTypeInt64: {
      auto alpha_value = inputs[kIndex2]->GetValueWithCheck<int64_t>();
      MAKE_SCALAR(alpha_value, inputs[0]->dtype_id(), alpha_);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "SubExt only support bool, float32, float64 and int64, but got " << alpha_dtype_id;
  }
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], alpha_, outputs[kIndex0]);
}

bool SubExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], alpha_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SubExt, SubExtAscend);
}  // namespace kernel
}  // namespace mindspore
