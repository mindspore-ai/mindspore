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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/searchsorted_aclnn_kernel.h"
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "kernel/ascend/aclnn/common/searchsorted_validate_utils.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace searchsorted {
namespace {
constexpr auto kSearchSortedOpName = "SearchSorted";

void CheckSearchSortedAclnnInputs(const std::vector<KernelTensor *> &inputs) {
  auto sorted_sequence_shape = inputs[kIndex0]->GetShapeVector();
  auto values_shape = inputs[kIndex1]->GetShapeVector();
  CheckSearchsortedInputShapes(sorted_sequence_shape, values_shape, kSearchSortedOpName);

  auto sorter = inputs[kIndex2];
  MS_EXCEPTION_IF_NULL(sorter);
  if (sorter->type_id() == kMetaTypeNone) {
    return;
  }

  CheckSearchsortedSorterType(sorter->dtype_id(), kSearchSortedOpName);
  CheckSearchsortedSorterShape(sorted_sequence_shape, sorter->GetShapeVector(), kSearchSortedOpName);
}
}  // namespace

void SearchsortedAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  CheckSearchSortedAclnnInputs(inputs);
  auto dtype = static_cast<TypeId>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]));
  auto out_int32 = false;
  if (dtype == kNumberTypeInt32) {
    out_int32 = true;
  }

  auto right = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex4]);

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], out_int32, right, inputs[kIndex2], outputs[kIndex0]);
}

bool SearchsortedAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto dtype = static_cast<TypeId>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]));
  auto out_int32 = false;
  if (dtype == kNumberTypeInt32) {
    out_int32 = true;
  }

  auto right = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex4]);

  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], out_int32, right, inputs[kIndex2], outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SearchSorted, SearchsortedAscend);
}  // namespace searchsorted
}  // namespace kernel
}  // namespace mindspore
