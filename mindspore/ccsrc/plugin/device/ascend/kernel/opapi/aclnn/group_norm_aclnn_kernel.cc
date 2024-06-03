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
#include "plugin/device/ascend/kernel/opapi/aclnn/group_norm_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNumberTwo = 2;
}  // namespace

void GroupNormAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  num_groups_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  auto eps_dtype_id = inputs[kIndex4]->dtype_id();
  switch (eps_dtype_id) {
    case kNumberTypeFloat32: {
      eps_ = static_cast<double>(inputs[kIndex4]->GetValueWithCheck<float>());
      break;
    }
    case kNumberTypeFloat64: {
      eps_ = inputs[kIndex4]->GetValueWithCheck<double>();
      break;
    }
    default:
      break;
  }
  const auto &x_shape = inputs[0]->GetShapeVector();
  n_ = x_shape[0];
  c_ = x_shape[1];
  hw_ = (x_shape.size() == kNumberTwo)
          ? 1
          : std::accumulate(x_shape.begin() + 2, x_shape.end(), 1, std::multiplies<int64_t>());

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex2], inputs[kIndex3], n_, c_, hw_, num_groups_, eps_,
                        outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
}

bool GroupNormAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex2], inputs[kIndex3], n_, c_, hw_, num_groups_, eps_,
        outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GroupNorm, GroupNormAscend);
}  // namespace kernel
}  // namespace mindspore
