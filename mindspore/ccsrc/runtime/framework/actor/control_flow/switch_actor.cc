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

#include "runtime/framework/actor/control_flow/switch_actor.h"
#include "runtime/framework/actor/control_flow/gather_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/include/async/async.h"
#include "abstract/utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void SwitchActor::Init() {
  // Init output data.
  output_data_.resize(output_branch_data_arrows_.size());
  for (size_t i = 0; i < output_branch_data_arrows_.size(); ++i) {
    auto &output_branch_data_arrows = output_branch_data_arrows_[i];
    auto &output_data = output_data_[i];
    for (auto &data_arrow : output_branch_data_arrows) {
      MS_EXCEPTION_IF_NULL(data_arrow);
      auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
      (void)output_data.emplace_back(std::move(data));
    }
  }
}

size_t SwitchActor::GetIndex(const OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  const auto &nodes_iter = input_nodes_.find(context->sequential_num_);
  if (nodes_iter == input_nodes_.end()) {
    MS_LOG(ERROR) << "Cannot find input node for switch actor:" << GetAID();
    return 0;
  }
  const auto &index_iter = nodes_iter->second.find(0);
  if (index_iter == nodes_iter->second.end() || index_iter->second.empty()) {
    MS_LOG(ERROR) << "Cannot find index input node for switch actor:" << GetAID();
    return 0;
  }

  const auto &index_node_with_index = index_iter->second[0];
  const auto &index_node = index_node_with_index.first;
  MS_EXCEPTION_IF_NULL(index_node);
  MS_EXCEPTION_IF_NULL(index_node->kernel_info());
  if (!AnfAlgo::OutputAddrExist(index_node, index_node_with_index.second, false)) {
    MS_LOG(ERROR) << "Invalid output index:" << index_node_with_index.second
                  << " for node:" << index_node->DebugString();
    return 0;
  }

  DeviceTensor *device_tensor = AnfAlgo::GetMutableOutputAddr(index_node, index_node_with_index.second, false).get();
  MS_EXCEPTION_IF_NULL(device_tensor);
  TypeId type_id = AnfAlgo::GetOutputInferDataType(index_node, index_node_with_index.second);
  size_t size = abstract::TypeIdSize(type_id);
  if (size > sizeof(int64_t)) {
    MS_LOG(ERROR) << "Index must be Int type.";
    return 0;
  }

  int64_t index = 0;
  char buf[kMaxSwitchCondSize] = {0};
  ShapeVector host_shape;
  if (!device_tensor->SyncDeviceToHost(host_shape, size, type_id, static_cast<void *>(buf))) {
    MS_LOG(ERROR) << GetAID().Name() << " get index from device address failed, type id:" << std::to_string(type_id)
                  << ", device type:" << std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceAddressType()));
    return 0;
  }

  if (type_id == TypeId::kNumberTypeInt32) {
    index = static_cast<int64_t>((static_cast<int32_t *>(static_cast<void *>(buf)))[0]);
  } else if (type_id == TypeId::kNumberTypeInt64) {
    index = (static_cast<int64_t *>(static_cast<void *>(buf)))[0];
  } else if (type_id == TypeId::kNumberTypeBool) {
    bool cond = (static_cast<bool *>(static_cast<void *>(buf)))[0];
    index = static_cast<int64_t>(cond ? 1 : 0);
  } else {
    MS_LOG(ERROR) << "Index must be Int type.";
    return 0;
  }

  // SwitchLayer node support negative index range [-size, -1].
  if (index < 0) {
    index += SizeToInt(input_result_num_ - 1);
  }
  return static_cast<size_t>(index);
}
}  // namespace runtime
}  // namespace mindspore
