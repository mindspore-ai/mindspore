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
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/include/async/async.h"
#include "abstract/utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void SwitchActor::Init() {
  // Init output data.
  output_data_.resize(output_branch_arrows_.size());
  for (size_t i = 0; i < output_branch_arrows_.size(); ++i) {
    auto &output_branch_arrow = output_branch_arrows_[i];
    auto &output_data = output_data_[i];
    for (auto &data_arrow : output_branch_arrow) {
      MS_EXCEPTION_IF_NULL(data_arrow);
      auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
      output_data.emplace_back(std::move(data));
    }
  }
}

void SwitchActor::RunOpData(OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  input_op_datas_[sequential_num].emplace_back(input_data);

  // When all the inputs are collected, then allocate memory and callback launch.
  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    EraseInput(context);
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
}

void SwitchActor::InitPartial(const AnfNodePtr &node, const size_t branch_id) {
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    CNodePtr cnode = node->cast<CNodePtr>();

    // The inputs of the Partial node is:
    // [0] ValueNode<Primitive> kPartial.
    // [1] ValueNode<FuncGraphPtr>.
    // [2..] Inputs.
    const auto &node_inputs = cnode->inputs();
    if (node_inputs.size() <= kPartialFuncGraphPos) {
      MS_LOG(EXCEPTION) << "Invalid Partial node:" << AnfAlgo::GetNodeDebugString(cnode);
    }

    const auto &func_graph = GetValueNode<FuncGraphPtr>(node_inputs[kPartialFuncGraphPos]);
    if (func_graph->output()->isa<ValueNode>()) {
      AddInput(func_graph->output(), branch_id);
      return;
    }

    branch_func_graph_[branch_id] = func_graph;
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
  output_branch_result_arrows_.resize(kSwitchPartialNum);
  output_branch_control_arrows_.resize(kSwitchPartialNum);

  input_nodes_.push_back(inputs[kSwitchCondPos]);
  input_datas_num_++;
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
  branch_func_graph_.resize(branch_nodes.size() - 1);
  output_branch_arrows_.resize(branch_nodes.size() - 1);
  output_branch_result_arrows_.resize(branch_nodes.size() - 1);
  output_branch_control_arrows_.resize(branch_nodes.size() - 1);

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

size_t SwitchActor::FetchDataNodePosition(const AnfNodePtr &data_node) const {
  const auto &iter = find(input_nodes_.begin(), input_nodes_.end(), data_node);
  if (iter == input_nodes_.end()) {
    MS_LOG(EXCEPTION) << "Data node: " << data_node->fullname_with_scope()
                      << " is not exist in gather actor:" << GetAID();
  }
  return iter - input_nodes_.begin();
}

void SwitchActor::AddInput(const AnfNodePtr &node, const size_t branch) {
  branch_total_inputs_[branch].push_back(node);

  if (node->isa<ValueNode>() && (!HasAbstractMonad(node))) {
    device_tensor_store_keys_.push_back({input_nodes_.size(), node.get()});
    branch_inputs_pos_[branch].push_back(input_nodes_.size());
    input_nodes_.push_back(node);
    return;
  }

  // Switch actor only receives parameter, updatestate node output is U, need to be skipped.
  if (IsPersistentDeviceTensor(node) || AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState)) {
    return;
  }

  auto iter = find(input_nodes_.begin(), input_nodes_.end(), node);
  if (iter == input_nodes_.end()) {
    branch_inputs_pos_[branch].push_back(input_nodes_.size());
    input_nodes_.push_back(node);
    ++input_datas_num_;
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
  input_device_tensors_.resize(input_nodes_.size());
  auto data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      input_device_tensors_[input_data->index_] = input_data->data_;
    }
  }
  data_iter->second.clear();

  for (const auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto device_tensor =
      DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second, device_context_->GetDeviceAddressType());
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_context_->GetDeviceAddressType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[device_tensor_store_key.first] = device_tensor;
  }
}

void SwitchActor::SendOutput(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto index = GetIndex();
  if (index >= output_branch_arrows_.size()) {
    MS_LOG(EXCEPTION) << "Switch actor invalid index:" << index;
  }

  auto &output_branch_arrow = output_branch_arrows_[index];
  auto &output_data = output_data_[index];
  for (size_t i = 0; i < output_branch_arrow.size(); ++i) {
    auto &data_arrow = output_branch_arrow[i];
    auto &data = output_data[i];
    MS_EXCEPTION_IF_NULL(data_arrow);
    MS_EXCEPTION_IF_NULL(data);
    data->data_ = input_device_tensors_[data_arrow->from_output_index_];
    Async(data_arrow->to_op_id_, &OpActor::RunOpData, data.get(), context);
  }

  // Send result.
  auto &output_branch_result_arrow = output_branch_result_arrows_[index];
  for (size_t i = 0; i < output_branch_result_arrow.size(); ++i) {
    auto &result_arrow = output_branch_result_arrow[i];
    MS_EXCEPTION_IF_NULL(result_arrow);
    size_t from_index = result_arrow->from_output_index_;
    for (const auto &backend_node : front_to_backend_parameter_[from_index]) {
      if (AnfAlgo::GetMutableOutputAddr(backend_node.first, backend_node.second).get() ==
          input_device_tensors_[from_index]) {
        Async(result_arrow->to_op_id_, &OutputActor::CollectOutput, backend_node.first, backend_node.second,
              result_arrow->to_input_index_, context);
        break;
      }
    }
  }

  // Send output control.
  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_branch_control_arrows_[index]) {
    Async(output_control, &OpActor::RunOpControl, source_aid, context);
  }
}

void SwitchActor::EraseInput(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    auto ret = input_op_datas_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input data failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  if (input_controls_num_ != 0) {
    auto ret = input_op_controls_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input controls failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }
}

void SwitchActor::SendMemoryFreeReq(OpContext<DeviceTensor> *context) {
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_device_tensors_, device_context_, context);
}

void SwitchActor::FetchInputNode(const std::vector<AnfNodePtr> &origin_parameters_order,
                                 const FrontToBackendNodeWithContext &front_to_backend_parameters,
                                 const std::unordered_map<AnfNodePtr, AnfNodePtr> &front_to_backend_kernel) {
  front_to_backend_parameter_.resize(input_nodes_.size());

  for (size_t i = 0; i < input_nodes_.size(); ++i) {
    const auto &input_node = input_nodes_[i];
    if (input_node->isa<ValueNode>()) {
      front_to_backend_parameter_[i].push_back({input_node, 0});
    } else if (input_node->isa<Parameter>()) {
      if (front_to_backend_parameters.find(input_node) != front_to_backend_parameters.end()) {
        const auto backend_node = front_to_backend_parameters.at(input_node).first;
        front_to_backend_parameter_[i].push_back({backend_node, 0});
      }
    } else if (input_node->isa<CNode>()) {
      if (IsCallNode(input_node)) {
        const auto func_graphs = FetchFuncGraphbyCallNode(input_node->cast<CNodePtr>());
        for (const auto func_graph : func_graphs) {
          if (func_graph->output()->isa<ValueNode>()) {
            front_to_backend_parameter_[i].push_back({func_graph->output(), 0});
          }
        }
      } else {
        const auto &kernel_with_index = AnfAlgo::VisitKernelWithReturnType(input_node, 0);
        if (front_to_backend_kernel.find(input_node) != front_to_backend_kernel.end()) {
          front_to_backend_parameter_[i].emplace_back(kernel_with_index);
        }
      }
    }
  }
}

}  // namespace runtime
}  // namespace mindspore
