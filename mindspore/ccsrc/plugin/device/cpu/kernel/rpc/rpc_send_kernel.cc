/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
bool RpcSendKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &) {
  // The workspace's memory size changes if is_dynamic_shape_.
  is_dynamic_shape_ = std::any_of(inputs.begin(), inputs.end(),
                                  [](const auto &kernel_tensor) { return kernel_tensor->IsDynamicShape(); });
  AssignWorkspaceSize(inputs);
  return true;
}

int RpcSendKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  // After Resize, we still need to process the workspace size list so that it won't be empty.
  AssignWorkspaceSize(inputs);
  return ret;
}

std::vector<KernelAttr> RpcSendKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

size_t RpcSendKernelMod::GetDynamicShapeMsgSize(const KernelTensor *dynamic_shape_input) {
  MS_EXCEPTION_IF_NULL(dynamic_shape_input);
  size_t msg_size = 0;
  auto shapes = dynamic_shape_input->GetShapeVector();
  TypeId data_type = dynamic_shape_input->dtype_id();
  size_t input_size = dynamic_shape_input->IsDynamicShape() ? kSizeZero : dynamic_shape_input->size();

  runtime::rpc::DynamicShapeMessage pb_msg;
  pb_msg.set_type_id(static_cast<int>(data_type));
  *pb_msg.mutable_shape_vector() = {shapes.begin(), shapes.end()};
  std::string pb_msg_str = pb_msg.SerializeAsString();

  msg_size += strlen(kRpcDynamicShapeData);
  msg_size += sizeof(size_t);
  msg_size += pb_msg_str.size();
  msg_size += input_size;
  return msg_size;
}

void RpcSendKernelMod::AssignWorkspaceSize(const std::vector<KernelTensor *> &inputs) {
  // Assign one piece of workspace memory with the same size of all inputs. It's the data which will be sent to remote.
  // Only allocate one piece of workspace memory to avoid extra memory copying and serialize inputs data to one message.
  workspace_size_list_.clear();
  size_t total_size = 0;
  total_size = std::accumulate(inputs.begin(), inputs.end(), total_size,
                               [this](size_t total_size, const KernelTensor *input_tensor) {
                                 return is_dynamic_shape_ ? (total_size + GetDynamicShapeMsgSize(input_tensor))
                                                          : (total_size + input_tensor->size());
                               });

  workspace_size_list_.push_back(total_size);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RpcSend, RpcSendKernelMod);
}  // namespace kernel
}  // namespace mindspore
