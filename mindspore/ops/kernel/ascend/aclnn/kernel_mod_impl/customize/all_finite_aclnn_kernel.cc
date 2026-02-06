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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/all_finite_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace all_finite {
namespace {
static constexpr size_t kAlignSize = 512;
}  // namespace

void AllFiniteAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  ws_mark_.clear();
  all_hash_id_.clear();
  std::vector<size_t> all_ws;
  for (auto input : inputs) {
    SetWorkspaceSizeList({});
    GetWorkspaceForResize(input, outputs[kIndex0]);
    (void)all_hash_id_.emplace_back(hash_id_);
    auto cur_ws = GetWorkspaceSizeList();
    if (cur_ws.empty()) {
      (void)ws_mark_.emplace_back(false);
    } else {
      (void)all_ws.emplace_back(cur_ws[0]);
      (void)ws_mark_.emplace_back(true);
    }
  }
  SetWorkspaceSizeList(all_ws);
}

bool AllFiniteAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto ret = CALL_ASCEND_API(aclrtMemsetAsync, outputs[kIndex0]->device_ptr(), kAlignSize, 0, kAlignSize, stream_ptr);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call runtime aclrtMemsetAsync error, ret[" << ret << "]";
    return false;
  }

  auto all_ws = GetWorkspaceSizeList();
  if (ws_mark_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Workspace size " << ws_mark_.size() << " is not equal inputs size  " << inputs.size();
  }
  if (all_hash_id_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << " Hash id size " << all_hash_id_.size() << " is not equal inputs size " << inputs.size();
  }
  size_t ws_index = 0;
  for (size_t i = 0; i < inputs.size(); i++) {
    hash_id_ = all_hash_id_[i];
    std::vector<KernelTensor *> cur_workspace;
    if (ws_mark_[i]) {
      SetWorkspaceSizeList({all_ws[ws_index]});
      (void)cur_workspace.emplace_back(workspace[ws_index]);
      ws_index++;
    } else {
      SetWorkspaceSizeList({});
    }
    RunOp(stream_ptr, cur_workspace, inputs[i], outputs[kIndex0]);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AllFinite, AllFiniteAscend);
}  // namespace all_finite
}  // namespace kernel
}  // namespace mindspore
