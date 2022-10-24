/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/rpc/rpc_send_kernel.h"
#include <string>
#include "runtime/device/ms_device_shape_transfer.h"
#include "proto/rpc.pb.h"

namespace mindspore {
namespace kernel {
void RpcSendKernelMod::Init(const CNodePtr &kernel_node) {
  DeprecatedNativeCpuKernelMod::Init(kernel_node);
  // Assign one piece of workspace memory with the same size of all inputs. It's the data which will be sent to remote.
  // Only allocate one piece of workspace memory to avoid extra memory copying and serialize inputs data to one message.
  size_t total_size = 0;
  if (common::AnfAlgo::IsDynamicShape(kernel_node)) {
    // In dynamic shape scenario, workspace size should be updated.
    size_t input_size = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_size; i++) {
      auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel_node, i, false);
      auto real_input = input_node_with_index.first;
      auto real_input_index = input_node_with_index.second;
      MS_EXCEPTION_IF_NULL(real_input);

      auto shapes = trans::GetRuntimePaddingShape(real_input, real_input_index);
      TypeId data_type = common::AnfAlgo::GetOutputInferDataType(real_input, real_input_index);

      runtime::rpc::DynamicShapeMessage pb_msg;
      pb_msg.set_type_id(static_cast<int>(data_type));
      *pb_msg.mutable_shape_vector() = {shapes.begin(), shapes.end()};
      std::string pb_msg_str = pb_msg.SerializeAsString();
      total_size += strlen(kRpcDynamicShapeData);
      total_size += sizeof(size_t);
      total_size += pb_msg_str.size();
      total_size += input_size_list_[i];
    }
  } else {
    total_size = std::accumulate(input_size_list_.begin(), input_size_list_.end(), total_size,
                                 [](size_t total_size, const auto &input_size) { return total_size + input_size; });
  }
  workspace_size_list_.push_back(total_size);
}

std::vector<KernelAttr> RpcSendKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RpcSend, RpcSendKernelMod);
}  // namespace kernel
}  // namespace mindspore
