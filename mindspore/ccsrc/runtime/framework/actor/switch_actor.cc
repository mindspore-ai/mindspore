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
#include "runtime/framework/actor/gather_actor.h"
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
  const auto &sequential_num = context->sequential_num_;
  auto &input_datas = input_data_[sequential_num];
  input_datas[input_data->index_].push(input_data->data_);

  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    EraseInput(context);
    SendOutput(context);
  }
}

void SwitchActor::RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  if (input_controls_[sequential_num].find(input_control) == input_controls_[sequential_num].end()) {
    input_controls_[sequential_num][input_control] = 0;
  }
  input_controls_[sequential_num][input_control]++;

  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    EraseInput(context);
    SendOutput(context);
  }
}

void SwitchActor::CollectBranchId(const int branch_id, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_branch_ids_[sequential_num].push(branch_id);
}

void SwitchActor::Initialize(const ControlNodeParserPtr &parser) {
  std::vector<AnfNodePtr> inputs = node_->inputs();

  if (IsPrimitive(inputs[0], prim::kPrimSwitch)) {
    InitSwitch();
  } else if (IsPrimitive(inputs[0], prim::kPrimReturn)) {
    InitReturn(parser);
  } else {
    InitSwitchLayer();
  }
  backend_parameters_.resize(input_nodes_.size());
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

void SwitchActor::InitVectorSize(const size_t num) {
  branch_inputs_pos_.resize(num);
  branch_func_graph_.resize(num);
  output_branch_arrows_.resize(num);
  output_branch_result_arrows_.resize(num);
  output_branch_control_arrows_.resize(num);
  output_branch_branch_arrows_.resize(num);
}

void SwitchActor::InitReturn(const ControlNodeParserPtr &parser) {
  const auto &func_graph = node_->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &call_num = parser->GetCallNumByFuncGraph(func_graph);
  InitVectorSize(call_num);
  AddCommonInput(func_graph->output());
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

  InitVectorSize(kSwitchPartialNum);

  const auto cond_node = AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchCondPos], 0);
  input_nodes_.push_back(cond_node);
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

  const auto cond_node = AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchLayerCondPos], 0);
  input_nodes_.push_back(cond_node);
  input_datas_num_++;

  // The second input of SwitchLayer is maketuple node, which includes all branches.
  auto branch_nodes = inputs[kSwitchLayerBranchPos]->cast<CNodePtr>()->inputs();
  InitVectorSize(branch_nodes.size() - 1);

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
  const auto data_node_with_index = AnfAlgo::VisitKernelWithReturnType(data_node, 0);
  const auto &iter = find(input_nodes_.begin(), input_nodes_.end(), data_node_with_index);
  if (iter == input_nodes_.end()) {
    MS_LOG(EXCEPTION) << "Data node: " << AnfAlgo::GetNodeDebugString(data_node)
                      << " is not exist in switch actor:" << GetAID();
  }
  return iter - input_nodes_.begin();
}

void SwitchActor::AddInput(const KernelWithIndex node_with_index, const size_t branch) {
  const auto &node = node_with_index.first;

  // Add weight and value node.
  if ((AnfAlgo::CheckPrimitiveType(node_, prim::kPrimReturn) && HasAbstractRef(node)) || node->isa<ValueNode>()) {
    const auto iter = find(input_nodes_.begin(), input_nodes_.end(), node_with_index);
    if (iter != input_nodes_.end()) {
      branch_inputs_pos_[branch].push_back(iter - input_nodes_.begin());
      return;
    }
    device_tensor_store_keys_.push_back({input_nodes_.size(), node.get()});
    branch_inputs_pos_[branch].push_back(input_nodes_.size());
    input_nodes_.push_back(node_with_index);
    return;
  }

  // Output of updatestate node is U, need to be skipped.
  if (HasAbstractRef(node)) {
    return;
  }

  // Add parameter.
  auto iter = find(input_nodes_.begin(), input_nodes_.end(), node_with_index);
  if (iter == input_nodes_.end()) {
    branch_inputs_pos_[branch].push_back(input_nodes_.size());
    input_nodes_.push_back(node_with_index);
    ++input_datas_num_;
  } else {
    branch_inputs_pos_[branch].push_back(iter - input_nodes_.begin());
  }
}

void SwitchActor::AddInput(const AnfNodePtr &node, const size_t branch) {
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState) || HasAbstractMonad(node)) {
    return;
  }

  const auto &real_input = AnfAlgo::VisitKernelWithReturnType(node, 0);

  if (AnfAlgo::CheckPrimitiveType(real_input.first, prim::kPrimMakeTuple)) {
    const auto &inputs = real_input.first->cast<CNodePtr>()->inputs();
    for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
      AddInput(inputs[i], branch);
    }
  } else if (IsCallNode(real_input.first)) {
    std::vector<AnfNodePtr> call_nodes;
    const auto call_output_num = FetchOutputSizebyCallNode(real_input.first, &call_nodes);

    if (call_output_num <= 0) {
      MS_LOG(EXCEPTION) << "Invalid output num for call input:" << AnfAlgo::GetNodeDebugString(real_input.first);
    }
    for (size_t i = 0; i < call_output_num; ++i) {
      AddInput({real_input.first, i}, branch);
    }
  } else {
    AddInput(real_input, branch);
  }
}

size_t SwitchActor::GetIndex(OpContext<DeviceTensor> *context) {
  if (need_branch_id_input_) {
    if (input_branch_ids_.find(context->sequential_num_) == input_branch_ids_.end() ||
        input_branch_ids_[context->sequential_num_].empty()) {
      MS_LOG(EXCEPTION) << "Invalid branch id for actor:" << GetAID();
    }
    size_t branch_id = input_branch_ids_[context->sequential_num_].top();
    input_branch_ids_[context->sequential_num_].pop();
    if (branch_id_to_index_.find(branch_id) == branch_id_to_index_.end()) {
      MS_LOG(EXCEPTION) << "Invalid branch id for switch actor:" << GetAID() << " branch id:" << branch_id;
    }
    return branch_id_to_index_[branch_id];
  }

  DeviceTensor *device_tensor = input_device_tensors_[0];
  if (device_tensor == nullptr) {
    MS_LOG(EXCEPTION) << "Index of switch actor is empty:" << GetAID();
  }
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
    auto data_iter = input_data_.find(context->sequential_num_);
    if (data_iter == input_data_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_datas_num_) {
      return false;
    }
    if (std::any_of(data_iter->second.begin(), data_iter->second.end(),
                    [](const auto &input_stack) { return input_stack.second.empty(); })) {
      return false;
    }
  }

  if (input_controls_num_ != 0) {
    auto data_iter = input_controls_.find(context->sequential_num_);
    if (data_iter == input_controls_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_controls_num_) {
      return false;
    }
    if (std::any_of(data_iter->second.begin(), data_iter->second.end(),
                    [](const auto &input_stack) { return input_stack.second == 0; })) {
      return false;
    }
  }

  return true;
}

void SwitchActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  input_device_tensors_.resize(input_nodes_.size());
  auto data_iter = input_data_.find(context->sequential_num_);
  if (data_iter != input_data_.end()) {
    for (auto &input_data : data_iter->second) {
      input_device_tensors_[input_data.first] = input_data.second.top();
      input_data.second.pop();
    }
  }

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

  auto control_iter = input_controls_.find(context->sequential_num_);
  if (control_iter != input_controls_.end()) {
    for_each(control_iter->second.begin(), control_iter->second.end(),
             [](auto &input_control) { input_control.second--; });
  }
}

void SwitchActor::SendOutput(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto index = GetIndex(context);
  if (index >= output_branch_arrows_.size()) {
    MS_LOG(EXCEPTION) << "Switch actor invalid index:" << index;
  }

  if (local_branch_id_ >= 0) {
    const auto &branch_arrows = output_branch_branch_arrows_[index];
    for (const auto &branch_arrow : branch_arrows) {
      Async(branch_arrow, &GatherActor::CollectBranchId, local_branch_id_, context);
    }
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
    for (const auto &backend_node : backend_parameters_[from_index]) {
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
  auto data_iter = input_data_.find(context->sequential_num_);
  if (data_iter != input_data_.end() && std::all_of(data_iter->second.begin(), data_iter->second.end(),
                                                    [](const auto &input_data) { return input_data.second.empty(); })) {
    auto ret = input_data_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input data failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  if (input_controls_num_ != 0) {
    auto control_iter = input_controls_.find(context->sequential_num_);
    if (control_iter != input_controls_.end() &&
        std::all_of(control_iter->second.begin(), control_iter->second.end(),
                    [](const auto &input_control) { return input_control.second == 0; })) {
      auto ret = input_controls_.erase(context->sequential_num_);
      if (ret == 0) {
        std::string error_info = "Erase input control failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
    }
  }
}

void SwitchActor::SendMemoryFreeReq(OpContext<DeviceTensor> *context) {
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_device_tensors_, device_context_, context);
}

void SwitchActor::FetchInputNode(const ControlNodeParserPtr &parser) {
  for (size_t i = 0; i < input_nodes_.size(); ++i) {
    const auto &input_node = input_nodes_[i].first;
    if (!HasAbstractRef(input_node)) {
      backend_parameters_[i] = parser->FetchBackendInputNodeByFrontNode(input_node);
      continue;
    }

    const auto &backend_weight = parser->FetchBackendNodebyWeightNode(input_node);
    if (backend_weight == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot find backend node for weight node:" << AnfAlgo::GetNodeDebugString(input_node);
    }
    backend_parameters_[i].push_back({backend_weight, 0});
  }
}
}  // namespace runtime
}  // namespace mindspore
