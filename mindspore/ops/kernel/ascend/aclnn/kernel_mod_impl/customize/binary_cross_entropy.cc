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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/binary_cross_entropy.h"
#include <vector>
#include <unordered_map>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
void BinaryCrossEntropyAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                        const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();

  auto reduction_imm = static_cast<Reduction>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]));
  std::unordered_map<Reduction, int64_t> reduction_map = {
    {Reduction::REDUCTION_SUM, 2}, {Reduction::MEAN, 1}, {Reduction::NONE, 0}};
  auto iter = reduction_map.find(reduction_imm);
  if (iter == reduction_map.end()) {
    MS_LOG(EXCEPTION) << "For BinaryCrossEntropy, the value of reduction is invalid.";
  }
  reduction_ = iter->second;

  const auto &input_shape = inputs[kIndex0]->GetShapeVector();
  const auto &weight_shape = inputs[kIndex2]->GetShapeVector();

  auto *weight = inputs[kIndex2];
  need_expand_weight_ = (!weight_shape.empty() && weight_shape != input_shape);
  if (need_expand_weight_) {
    input_shape_ = input_shape;
    weight_expand_.SetType(inputs[kIndex2]->GetType());
    weight_expand_.SetShape(std::make_shared<abstract::TensorShape>(input_shape));
    weight = &weight_expand_;
    GetWorkspaceForResizeExpandWeight(inputs[kIndex2], input_shape, &weight_expand_);
  }

  GetWorkspaceForResizeBCE(inputs[kIndex0], inputs[kIndex1], weight, reduction_, outputs[kIndex0]);
  if (need_expand_weight_) {
    const auto output_size =
      ops::CalOutputSize(input_shape, mindspore::abstract::TypeIdSize(inputs[kIndex2]->dtype_id()));
    workspace_size_list_.emplace_back(output_size);
  }
}

bool BinaryCrossEntropyAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto *weight = inputs[kIndex2];
  if (need_expand_weight_) {
    // Expand workspace is placed after the main op workspace, at the last slot.
    size_t ws_offset = workspace.size() - 1;
    weight_expand_.set_device_ptr(workspace[ws_offset]->device_ptr());
    weight = &weight_expand_;
    RunOpExpandWeight(stream_ptr, workspace, inputs[kIndex2], input_shape_, &weight_expand_);
  }

  RunOpBCE(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], weight, reduction_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BinaryCrossEntropy, BinaryCrossEntropyAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
