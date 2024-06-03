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
#include "plugin/device/ascend/kernel/opapi/aclnn/grouped_matmul_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputXIdx = 0;
constexpr size_t kInputWeightIdx = 1;
constexpr size_t kInputBiasIdx = 2;
constexpr size_t kInputScaleIdx = 3;
constexpr size_t kInputOffsetIdx = 4;
constexpr size_t kInputAntiquantScaleIdx = 5;
constexpr size_t kInputAntiquantOffsetIdx = 6;
}  // namespace
void GroupedMatmulAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  std::vector<int64_t> group_info = GetValue<std::vector<int64_t>>(primitive_->GetAttr("group_info"));

  std::vector<int64_t> dyn_input_idx{};
  int64_t idx = 0;
  for (size_t i = 0; i < group_info.size(); ++i) {
    idx += (group_info[i] == 0 ? 1 : group_info[i]);
    (void)dyn_input_idx.emplace_back(idx);
  }

  std::vector<KernelTensor *> x;
  std::vector<KernelTensor *> weight;
  std::vector<KernelTensor *> bias;
  std::vector<KernelTensor *> scale;
  std::vector<KernelTensor *> offset;
  std::vector<KernelTensor *> antiquant_scale;
  std::vector<KernelTensor *> antiquant_offset;

  x.assign(inputs.begin(), inputs.begin() + dyn_input_idx[kInputXIdx]);
  weight.assign(inputs.begin() + dyn_input_idx[kInputXIdx], inputs.begin() + dyn_input_idx[kInputWeightIdx]);
  bias.assign(inputs.begin() + dyn_input_idx[kInputWeightIdx], inputs.begin() + dyn_input_idx[kInputBiasIdx]);
  scale.assign(inputs.begin() + dyn_input_idx[kInputBiasIdx], inputs.begin() + dyn_input_idx[kInputScaleIdx]);
  offset.assign(inputs.begin() + dyn_input_idx[kInputScaleIdx], inputs.begin() + dyn_input_idx[kInputOffsetIdx]);
  antiquant_scale.assign(inputs.begin() + dyn_input_idx[kInputOffsetIdx],
                         inputs.begin() + dyn_input_idx[kInputAntiquantScaleIdx]);
  antiquant_offset.assign(inputs.begin() + dyn_input_idx[kInputAntiquantScaleIdx],
                          inputs.begin() + dyn_input_idx[kInputAntiquantOffsetIdx]);
  auto group_list_tensor = *(inputs.end() - kIndex3);
  MS_EXCEPTION_IF_NULL(group_list_tensor);

  auto split_item_tensor = *(inputs.end() - kIndex2);
  MS_EXCEPTION_IF_NULL(split_item_tensor);
  int64_t split_item = split_item_tensor->GetValueWithCheck<int64_t>();

  auto group_type_tensor = *(inputs.end() - kIndex1);
  MS_EXCEPTION_IF_NULL(group_type_tensor);
  int64_t group_type = group_type_tensor->GetValueWithCheck<int64_t>();

  GetWorkspaceForResize(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list_tensor,
                        split_item, group_type, outputs);
}

bool GroupedMatmulAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  std::vector<int64_t> group_info = GetValue<std::vector<int64_t>>(primitive_->GetAttr("group_info"));

  std::vector<int64_t> dyn_input_idx{};
  int64_t idx = 0;
  for (size_t i = 0; i < group_info.size(); ++i) {
    idx += (group_info[i] == 0 ? 1 : group_info[i]);
    (void)dyn_input_idx.emplace_back(idx);
  }

  std::vector<KernelTensor *> x;
  std::vector<KernelTensor *> weight;
  std::vector<KernelTensor *> bias;
  std::vector<KernelTensor *> scale;
  std::vector<KernelTensor *> offset;
  std::vector<KernelTensor *> antiquant_scale;
  std::vector<KernelTensor *> antiquant_offset;

  x.assign(inputs.begin(), inputs.begin() + dyn_input_idx[kInputXIdx]);
  weight.assign(inputs.begin() + dyn_input_idx[kInputXIdx], inputs.begin() + dyn_input_idx[kInputWeightIdx]);
  bias.assign(inputs.begin() + dyn_input_idx[kInputWeightIdx], inputs.begin() + dyn_input_idx[kInputBiasIdx]);
  scale.assign(inputs.begin() + dyn_input_idx[kInputBiasIdx], inputs.begin() + dyn_input_idx[kInputScaleIdx]);
  offset.assign(inputs.begin() + dyn_input_idx[kInputScaleIdx], inputs.begin() + dyn_input_idx[kInputOffsetIdx]);
  antiquant_scale.assign(inputs.begin() + dyn_input_idx[kInputOffsetIdx],
                         inputs.begin() + dyn_input_idx[kInputAntiquantScaleIdx]);
  antiquant_offset.assign(inputs.begin() + dyn_input_idx[kInputAntiquantScaleIdx],
                          inputs.begin() + dyn_input_idx[kInputAntiquantOffsetIdx]);
  auto group_list_tensor = *(inputs.end() - kIndex3);
  MS_EXCEPTION_IF_NULL(group_list_tensor);

  auto split_item_tensor = *(inputs.end() - kIndex2);
  MS_EXCEPTION_IF_NULL(split_item_tensor);
  int64_t split_item = split_item_tensor->GetValueWithCheck<int64_t>();

  auto group_type_tensor = *(inputs.end() - kIndex1);
  MS_EXCEPTION_IF_NULL(group_type_tensor);
  int64_t group_type = group_type_tensor->GetValueWithCheck<int64_t>();

  RunOp(stream_ptr, workspace, x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list_tensor,
        split_item, group_type, outputs);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GroupedMatmul, GroupedMatmulAscend);
}  // namespace kernel
}  // namespace mindspore
