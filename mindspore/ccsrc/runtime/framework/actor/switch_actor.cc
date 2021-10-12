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
      (void)output_data.emplace_back(std::move(data));
    }
  }
}

void SwitchActor::RunOpData(OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *const context) {
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

void SwitchActor::CollectBranchId(const int branch_id, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_branch_ids_[sequential_num].push(branch_id);
}

void SwitchActor::ParseInput(const ControlNodeParserPtr &parser) {
  std::vector<AnfNodePtr> inputs = node_->inputs();

  if (IsPrimitive(inputs[0], prim::kPrimSwitch)) {
    ParseSwitchInput();
  } else if (IsPrimitive(inputs[0], prim::kPrimReturn)) {
    ParseReturnInput(parser);
  } else {
    ParseSwitchLayerInput();
  }
  backend_parameters_.resize(input_nodes_.size());
}

void SwitchActor::ParsePartialInput(const AnfNodePtr &node, const size_t branch_id) {
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    CNodePtr cnode = node->cast<CNodePtr>();

    // The inputs of the Partial node is:
    // [0] ValueNode<Primitive> kPartial.
    // [1] ValueNode<FuncGraphPtr>.
    // [2..] Inputs.
    auto partial_inputs = cnode->inputs();
    if (partial_inputs.size() <= kPartialFuncGraphPos) {
      MS_LOG(EXCEPTION) << "Invalid Partial node:" << AnfAlgo::GetNodeDebugString(cnode);
    }

    auto func_graph = GetValueNode<FuncGraphPtr>(partial_inputs[kPartialFuncGraphPos]);

    branch_func_graph_[branch_id] = func_graph;
    for (size_t j = kPartialInputStartPos; j < partial_inputs.size(); ++j) {
      AddInput(partial_inputs[j], branch_id);
    }
  } else if (IsValueNode<FuncGraph>(node)) {
    const auto func_graph = GetValueNode<FuncGraphPtr>(node);
    branch_func_graph_[branch_id] = func_graph;
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

void SwitchActor::ParseReturnInput(const ControlNodeParserPtr &parser) {
  const auto &func_graph = node_->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &call_num = parser->GetCallNumByFuncGraph(func_graph);
  InitVectorSize(call_num);

  AddCommonInput(func_graph->output());
}

void SwitchActor::ParseSwitchInput() {
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
  ParsePartialInput(inputs[kSwitchFalseBranchPos], static_cast<size_t>(false));
  ParsePartialInput(inputs[kSwitchTrueBranchPos], static_cast<size_t>(true));
}

void SwitchActor::ParseSwitchLayerInput() {
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
  for (size_t i = kMakeTupleInputStartPos; i < branch_nodes.size(); ++i) {
    if (AnfAlgo::CheckPrimitiveType(branch_nodes[i], prim::kPrimPartial)) {
      ParsePartialInput(branch_nodes[i], i - kMakeTupleInputStartPos);
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

  // The value node and weight node need to be placed in the device store. The switch actor has three inputs:
  // 1) The input of the switch is the value node.
  // 2) There is a weight node or value node in the return of the sub funcgraph.
  if ((AnfAlgo::CheckPrimitiveType(node_, prim::kPrimReturn) && node->isa<Parameter>() && HasAbstractRef(node)) ||
      node->isa<ValueNode>()) {
    const auto iter = find(input_nodes_.begin(), input_nodes_.end(), node_with_index);
    if (iter != input_nodes_.end()) {
      branch_inputs_pos_[branch].push_back(iter - input_nodes_.begin());
      return;
    }
    (void)device_tensor_store_keys_.emplace_back(input_nodes_.size(), node.get());
    branch_inputs_pos_[branch].push_back(input_nodes_.size());
    input_nodes_.push_back(node_with_index);
    return;
  }

  // Output of updatestate node is U, need to be skipped.
  if (node->isa<Parameter>() && HasAbstractRef(node)) {
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
    if (call_output_num == 0) {
      MS_LOG(EXCEPTION) << "Invalid output num for call input:" << AnfAlgo::GetNodeDebugString(real_input.first);
    }
    for (size_t i = 0; i < call_output_num; ++i) {
      AddInput({real_input.first, i}, branch);
    }
  } else if (real_input.first->isa<ValueNode>() && real_input.first->cast<ValueNodePtr>()->value()->isa<ValueTuple>()) {
    const auto &value = real_input.first->cast<ValueNodePtr>()->value();
    const auto &tuple_value = value->cast<ValueTuplePtr>();
    for (size_t i = 0; i < tuple_value->value().size(); ++i) {
      AddInput({real_input.first, i}, branch);
    }
  } else {
    AddInput(real_input, branch);
  }
}

size_t SwitchActor::GetIndex(const OpContext<DeviceTensor> *const context) {
  if (need_branch_id_input_) {
    if (input_branch_ids_.find(context->sequential_num_) == input_branch_ids_.end() ||
        input_branch_ids_[context->sequential_num_].empty()) {
      MS_LOG(ERROR) << "Invalid branch id for actor:" + GetAID().Name();
    }
    auto branch_id = input_branch_ids_[context->sequential_num_].top();
    input_branch_ids_[context->sequential_num_].pop();
    if (branch_id_to_index_.find(branch_id) == branch_id_to_index_.end()) {
      MS_LOG(ERROR) << "Invalid branch id for switch actor:" + GetAID().Name() +
                         " branch id:" + std::to_string(branch_id);
    }
    return branch_id_to_index_[branch_id];
  }

  DeviceTensor *device_tensor = input_device_tensors_[0];
  MS_EXCEPTION_IF_NULL(device_tensor);

  auto inputs = node_->inputs();
  TypeId type_id = AnfAlgo::GetOutputInferDataType(inputs[kSwitchCondPos], 0);
  size_t size = abstract::TypeIdSize(type_id);
  if (size > sizeof(int64_t)) {
    MS_LOG(ERROR) << "Index must be Int type.";
  }

  int64_t index = 0;
  char buf[kMaxSwitchCondSize] = {0};
  ShapeVector host_shape;
  if (!device_tensor->SyncDeviceToHost(host_shape, size, type_id, static_cast<void *>(buf))) {
    MS_LOG(ERROR) << GetAID().Name() << " get index from device address failed, type id:" << std::to_string(type_id)
                  << ", device type:" << std::to_string(static_cast<int>(device_context_->GetDeviceAddressType()));
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
  }

  // SwitchLayer node support negative index range [-size, -1].
  if (index < 0) {
    index += SizeToInt(branch_func_graph_.size());
  }
  return static_cast<size_t>(index);
}

bool SwitchActor::CheckLaunchCondition(OpContext<DeviceTensor> *const context) const {
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

void SwitchActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
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
    (void)for_each(control_iter->second.begin(), control_iter->second.end(),
                   [](auto &input_control) { input_control.second--; });
  }
}

void SwitchActor::SendOutput(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto index = GetIndex(context);
  if (index >= output_branch_arrows_.size()) {
    std::string error_info = "Switch actor:" + GetAID().Name() + " invalid index:" + std::to_string(index);
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  // Must be the execution order: send branch id --> send result --> send data --> send control, avoid the illegal
  // timing problem.
  // 1.Send branch id.
  if (local_branch_id_ >= 0) {
    const auto &branch_arrows = output_branch_branch_arrows_[index];
    for (const auto &branch_arrow : branch_arrows) {
      Async(branch_arrow, &GatherActor::CollectBranchId, local_branch_id_, context);
    }
  }

  // 2.Send result.
  auto &output_branch_result_arrow = output_branch_result_arrows_[index];
  for (size_t i = 0; i < output_branch_result_arrow.size(); ++i) {
    auto &result_arrow = output_branch_result_arrow[i];
    MS_EXCEPTION_IF_NULL(result_arrow);
    if (result_arrow->from_output_index_ >= SizeToInt(branch_inputs_pos_[index].size())) {
      std::string error_info =
        "Invalid from index in switch actor, from index:" + std::to_string(result_arrow->from_output_index_) +
        " total:" + std::to_string(branch_inputs_pos_[index].size()) + " actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    size_t from_index = branch_inputs_pos_[index][IntToSize(result_arrow->from_output_index_)];

    bool is_send = false;
    for (const auto &backend_node : backend_parameters_[from_index]) {
      for (size_t j = 0; j < AnfAlgo::GetOutputTensorNum(backend_node.first); ++j) {
        if (backend_node.first->kernel_info() != nullptr && AnfAlgo::OutputAddrExist(backend_node.first, j, false) &&
            AnfAlgo::GetMutableOutputAddr(backend_node.first, j, false).get() == input_device_tensors_[from_index]) {
          auto output_index = j;
          Async(result_arrow->to_op_id_, &OutputActor::CollectOutput, backend_node.first, output_index,
                result_arrow->to_input_index_, context);
          is_send = true;
          break;
        }
      }
    }
    if (!is_send) {
      std::string error_info = "Failed to get backend node of switch actor output, actor:" + GetAID().Name() +
                               " branch:" + std::to_string(index) +
                               " index:" + std::to_string(result_arrow->from_output_index_) + " output pos" +
                               std::to_string(branch_inputs_pos_[index][IntToSize(result_arrow->from_output_index_)]) +
                               " output index" + std::to_string(result_arrow->to_input_index_);
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  // 3.Send Data.
  auto &output_branch_arrow = output_branch_arrows_[index];
  auto &output_data = output_data_[index];
  for (size_t i = 0; i < output_branch_arrow.size(); ++i) {
    auto &data_arrow = output_branch_arrow[i];
    auto &data = output_data[i];
    MS_EXCEPTION_IF_NULL(data_arrow);
    MS_EXCEPTION_IF_NULL(data);
    data->data_ = input_device_tensors_[IntToSize(data_arrow->from_output_index_)];
    Async(data_arrow->to_op_id_, &OpActor::RunOpData, data.get(), context);
  }

  // 4.Send output control.
  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_branch_control_arrows_[index]) {
    Async(output_control, &OpActor::RunOpControl, source_aid, context);
  }
}

void SwitchActor::EraseInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto data_iter = input_data_.find(context->sequential_num_);
  if (data_iter != input_data_.end() && std::all_of(data_iter->second.begin(), data_iter->second.end(),
                                                    [](const auto &input_data) { return input_data.second.empty(); })) {
    auto ret = input_data_.erase(context->sequential_num_);
    if (ret == 0) {
      MS_LOG(WARNING) << "Erase input data failed for switch actor: " << GetAID();
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

void SwitchActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_device_tensors_, device_context_, context);
}

void SwitchActor::FetchInputNode(const ControlNodeParserPtr &parser) {
  for (size_t i = 0; i < input_nodes_.size(); ++i) {
    const auto &input_node = input_nodes_[i].first;
    if (!(input_node->isa<Parameter>() && HasAbstractRef(input_node))) {
      backend_parameters_[i] = parser->FetchBackendInputNodeByFrontNode(input_node);
      continue;
    }

    const auto &backend_weight = parser->FetchBackendNodebyWeightNode(input_node);
    if (backend_weight == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot find backend node for weight node:" << AnfAlgo::GetNodeDebugString(input_node);
    }
    (void)backend_parameters_[i].emplace(backend_weight, 0);
  }
}
}  // namespace runtime
}  // namespace mindspore
