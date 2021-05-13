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

#include "runtime/framework/actor/switch_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/include/async/async.h"
#include "abstract/utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
constexpr size_t kSwitchInputNum = 4;
constexpr size_t kSwitchCondPos = 1;
constexpr size_t kSwitchPartialNum = 2;
constexpr size_t kSwitchLayerCondPos = 1;
constexpr size_t kSwitchLayerBranchPos = 2;
constexpr size_t kSwitchLayerInputNum = 3;
constexpr size_t kMaxSwitchCondSize = 8;
constexpr size_t kSwitchTrueBranchPos = 2;
constexpr size_t kSwitchFalseBranchPos = 3;
constexpr size_t kPartialFuncGraphPos = 1;
constexpr size_t kPartialInputStartPos = 2;

void SwitchActor::RunOpData(OpDataPtr<DeviceTensor> input_data, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  input_op_datas_[sequential_num].emplace_back(input_data);

  // When all the inputs are collected, then allocate memory and callback launch.
  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    SendOutput(context);
  }
}

void SwitchActor::Initialize() {
  std::vector<AnfNodePtr> inputs = node_->inputs();

  if (IsPrimitive(inputs[0], prim::kPrimSwitch)) {
    InitSwitch();
  } else {
    InitSwitchLayer();
  }
  input_datas_num_ = input_nodes_.size();
}

void SwitchActor::InitPartial(const AnfNodePtr &node, const size_t branch_id) {
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    CNodePtr cnode = node->cast<CNodePtr>();

    // The inputs of the Partial node is:
    // [0] ValueNode<Primitive> kPartial.
    // [1] ValueNode<FuncGraphPtr>.
    // [2..] Inputs.
    auto node_inputs = cnode->inputs();
    branch_func_graph_[branch_id] = GetValueNode<FuncGraphPtr>(node_inputs[kPartialFuncGraphPos]);
    for (size_t j = kPartialInputStartPos; j < node_inputs.size(); ++j) {
      AddInput(node_inputs[j], branch_id);
    }
  } else {
    AddInput(node, branch_id);
  }
}

void SwitchActor::InitSwitch() {
  // The inputs of the switch node:
  // [0] ValueNode<Primitive> kSwitch.
  // [1] switch condition.
  // [2] Partial node: true branch.
  // [3] Partial node: false branch.
  std::vector<AnfNodePtr> inputs = node_->inputs();
  if (inputs.size() != kSwitchInputNum) {
    MS_LOG(EXCEPTION) << "Length of inputs of primitive " << prim::kPrimSwitch->name() << " is not equal 4";
  }

  branch_total_inputs_.resize(kSwitchPartialNum);
  branch_inputs_pos_.resize(kSwitchPartialNum);
  branch_func_graph_.resize(kSwitchPartialNum);
  output_branch_arrows_.resize(kSwitchPartialNum);
  input_nodes_.push_back(inputs[kSwitchCondPos]);

  // Init the two branches of switch node.
  InitPartial(inputs[kSwitchFalseBranchPos], static_cast<size_t>(false));
  InitPartial(inputs[kSwitchTrueBranchPos], static_cast<size_t>(true));
}

void SwitchActor::InitSwitchLayer() {
  // The inputs of the switch node:
  // [0] ValueNode<Primitive> kSwitchLayer.
  // [1] switchLayer index.
  // [2] MakeTuple node: tuple of branches.
  std::vector<AnfNodePtr> inputs = node_->inputs();
  if (inputs.size() != kSwitchLayerInputNum) {
    MS_LOG(EXCEPTION) << "Length of inputs of primitive " << prim::kPrimSwitchLayer->name() << " is not equal 3";
  }

  input_nodes_.push_back(inputs[kSwitchLayerCondPos]);

  // The second input of SwitchLayer is maketuple node, which includes all branches.
  auto branch_nodes = inputs[kSwitchLayerBranchPos]->cast<CNodePtr>()->inputs();
  branch_total_inputs_.resize(branch_nodes.size() - 1);
  branch_inputs_pos_.resize(branch_nodes.size() - 1);
  branch_device_tensor_store_keys_.resize(branch_nodes.size() - 1);
  branch_func_graph_.resize(branch_nodes.size() - 1);
  output_branch_arrows_.resize(branch_nodes.size() - 1);

  // Parse all branches.
  for (size_t i = 1; i < branch_nodes.size(); ++i) {
    if (AnfAlgo::CheckPrimitiveType(branch_nodes[i], prim::kPrimPartial)) {
      InitPartial(branch_nodes[i], i - 1);
    } else if (branch_nodes[i]->isa<ValueNode>()) {
      branch_func_graph_[i - 1] = GetValueNode<FuncGraphPtr>(branch_nodes[i]);
    }
  }
}

void SwitchActor::AddCommonInput(const AnfNodePtr &node) {
  for (size_t i = 0; i < branch_inputs_pos_.size(); ++i) {
    AddInput(node, i);
  }
}

void SwitchActor::AddInput(const AnfNodePtr &node, const size_t branch) {
  branch_total_inputs_[branch].push_back(node);
  if (IsPersistentDeviceTensor(node)) {
    return;
  }
  auto iter = find(input_nodes_.begin(), input_nodes_.end(), node);
  if (iter == input_nodes_.end()) {
    branch_inputs_pos_[branch].push_back(input_nodes_.size());
    input_nodes_.push_back(node);
  } else {
    branch_inputs_pos_[branch].push_back(iter - input_nodes_.begin());
  }
}

size_t SwitchActor::GetIndex() {
  DeviceTensor *device_tensor = input_device_tensors_[0];
  auto inputs = node_->inputs();
  TypeId type_id = AnfAlgo::GetOutputInferDataType(inputs[kSwitchCondPos], 0);
  size_t size = abstract::TypeIdSize(type_id);
  if (size > sizeof(int64_t)) {
    MS_LOG(EXCEPTION) << "Index must be Int type.";
  }

  int64_t index = 0;
  char buf[kMaxSwitchCondSize] = {0};
  ShapeVector host_shape;
  device_tensor->SyncDeviceToHost(host_shape, size, type_id, static_cast<void *>(buf));

  if (type_id == TypeId::kNumberTypeInt32) {
    index = static_cast<int64_t>((static_cast<int32_t *>(static_cast<void *>(buf)))[0]);
  } else if (type_id == TypeId::kNumberTypeInt64) {
    index = (static_cast<int64_t *>(static_cast<void *>(buf)))[0];
  } else if (type_id == TypeId::kNumberTypeBool) {
    bool cond = (static_cast<bool *>(static_cast<void *>(buf)))[0];
    index = static_cast<int64_t>(cond ? 1 : 0);
  } else {
    MS_LOG(EXCEPTION) << "Index must be Int type.";
  }

  // SwitchLayer node support negative index range [-size, -1].
  if (index < 0) {
    index += branch_func_graph_.size();
  }
  if (index > static_cast<int64_t>(SIZE_MAX)) {
    MS_LOG(EXCEPTION) << "Index is too large:" << index;
  }
  return static_cast<size_t>(index);
}

bool SwitchActor::CheckLaunchCondition(OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    auto data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_datas_num_) {
      return false;
    }
  }
  return true;
}

void SwitchActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto input_size = input_datas_num_ + branch_device_tensor_store_keys_.size();
  input_device_tensors_.resize(input_size);
  auto data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      input_device_tensors_[input_data->index_] = input_data->data_;
    }
  }
  data_iter->second.clear();

  for (auto &device_tensor_store_key : branch_device_tensor_store_keys_) {
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second);
    input_device_tensors_[device_tensor_store_key.first] = device_tensor.get();
  }
}

void SwitchActor::SendOutput(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto index = GetIndex();
  if (index >= output_branch_arrows_.size()) {
    MS_LOG(EXCEPTION) << "Switch actor invalid index:" << index;
  }
  for (const auto &arrow : output_branch_arrows_[index]) {
    auto device_address = input_device_tensors_[arrow->from_output_index_];
    auto data = std::make_shared<OpData<DeviceTensor>>(arrow->to_op_id_, device_address, arrow->to_input_index_);
    Async(arrow->to_op_id_, &OpActor::RunOpData, data, context);
  }
}

void SwitchActor::FreeMemory(OpContext<DeviceTensor> *context) {
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, input_device_tensors_, device_contexts_, context);
}

}  // namespace runtime
}  // namespace mindspore
