/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "runtime/pynative/op_runtime_info.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/utils.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace kernel {
void AscendKernelMod::SetAtomicCleanNodes(const std::vector<CNodePtr> &atomic_clean_node) {
  atomic_clean_nodes_.resize(atomic_clean_node.size());
  for (size_t i = 0; i < atomic_clean_node.size(); ++i) {
    atomic_clean_nodes_[i] = atomic_clean_node[i];
  }
}

void AscendKernelMod::UpdateOutputSizeList() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto op_runtime_info = node->user_data<runtime::OpRuntimeInfo>();
  auto cnode = node->cast<CNodePtr>();
  if (op_runtime_info != nullptr) {
    op_runtime_info->Resize(node);
  }
  for (size_t i = 0; i < output_size_list_.size(); ++i) {
    auto ori_output_size = output_size_list_[i];
    size_t real_output_size =
      (op_runtime_info == nullptr) ? AnfAlgo::GetOutputTensorMemSize(cnode, i) : op_runtime_info->output_tensor_size(i);
    if (ori_output_size != real_output_size) {
      output_size_list_[i] = real_output_size;
    }
  }
}

bool AscendKernelMod::IsNeedRetrieveOutputShape() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if (IsOneOfComputeDepend(op_name)) {
    is_need_retrieve_output_shape_ = true;
  }
  return is_need_retrieve_output_shape_;
}
}  // namespace kernel
}  // namespace mindspore
