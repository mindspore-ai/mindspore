/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/group_norm_grad_aclnn_kernel.h"
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
#include "mindspore/core/utils/shape_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNumberTwo = 2;
}  // namespace

void GroupNormGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  const auto &x_shape = inputs[kIndex1]->GetShapeVector();
  batch_ = x_shape[kIndex0];
  channel_ = x_shape[kIndex1];
  HxW_ = (x_shape.size() == kNumberTwo)
           ? 1
           : std::accumulate(x_shape.begin() + kIndex2, x_shape.end(), 1, std::multiplies<int64_t>());
  num_groups_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex5]);
  auto dx_is_require = static_cast<uint8_t>(transform::ConvertKernelTensor<bool>(inputs[kIndex6]));
  auto dgamma_is_require = static_cast<uint8_t>(transform::ConvertKernelTensor<bool>(inputs[kIndex7]));
  auto dbeta_is_require = static_cast<uint8_t>(transform::ConvertKernelTensor<bool>(inputs[kIndex8]));

  (void)output_mask_.emplace_back(dx_is_require);
  (void)output_mask_.emplace_back(dgamma_is_require);
  (void)output_mask_.emplace_back(dbeta_is_require);

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], batch_,
                        channel_, HxW_, num_groups_, output_mask_, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2]);
}

bool GroupNormGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
        batch_, channel_, HxW_, num_groups_, output_mask_, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GroupNormGrad, GroupNormGradAscend);
}  // namespace kernel
}  // namespace mindspore
