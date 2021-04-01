/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/session/kernel_graph.h"
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <set>
#include "base/core_ops.h"
#include "ir/param_info.h"
#include "utils/utils.h"
#include "utils/check_convert_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_info.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace session {
namespace {
constexpr auto kIsFeatureMapOutput = "IsFeatureMapOutput";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
constexpr size_t k5dDims = 5;
const std::set<std::string> kOpAssignKernelNameList = {prim::kPrimAssign->name(), prim::kPrimAssignAdd->name(),
                                                       prim::kPrimAssignSub->name()};

void PushNoVisitedNode(const AnfNodePtr &node, std::queue<AnfNodePtr> *que,
                       std::unordered_set<AnfNodePtr> *visited_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(que);
  MS_EXCEPTION_IF_NULL(visited_nodes);
  if (visited_nodes->find(node) == visited_nodes->end()) {
    que->push(node);
    (void)visited_nodes->insert(node);
    MS_LOG(DEBUG) << "Push que:" << node->DebugString();
  }
}

std::vector<AnfNodePtr> GetCallRealOutputs(const AnfNodePtr &call_node) {
  auto item_with_index =
    AnfAlgo::VisitKernelWithReturnType(call_node, 0, false, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple});
  AnfNodePtr node = item_with_index.first;
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    auto outputs = AnfAlgo::GetAllOutput(node);
    std::set<AnfNodePtr> memo;
    std::vector<AnfNodePtr> new_output;
    for (auto &output : outputs) {
      if (memo.find(output) != memo.end()) {
        continue;
      }
      memo.insert(output);
      new_output.push_back(output);
    }
    if (new_output.size() == 1 && AnfAlgo::CheckPrimitiveType(new_output[0], prim::kPrimCall)) {
      node = new_output[0];
    }
  }
  if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall)) {
    return {node};
  }
  std::vector<AnfNodePtr> real_inputs;
  auto child_graphs = AnfAlgo::GetCallSwitchKernelGraph(node->cast<CNodePtr>());
  for (const auto &child_graph : child_graphs) {
    auto real_input = child_graph->output();
    auto child_real_inputs = GetCallRealOutputs(real_input);
    std::copy(child_real_inputs.begin(), child_real_inputs.end(), std::back_inserter(real_inputs));
  }
  return real_inputs;
}

bool IsSameLabel(const CNodePtr &left, const CNodePtr &right) {
  if (left == right) {
    return true;
  }
  if (left == nullptr || right == nullptr) {
    return false;
  }
  if (!IsPrimitiveCNode(left, GetCNodePrimitive(right))) {
    return false;
  }
  if (AnfAlgo::HasNodeAttr(kAttrLabelIndex, left) && AnfAlgo::HasNodeAttr(kAttrLabelIndex, right)) {
    return AnfAlgo::GetNodeAttr<uint32_t>(left, kAttrLabelIndex) ==
           AnfAlgo::GetNodeAttr<uint32_t>(right, kAttrLabelIndex);
  }
  return false;
}

void SyncDeviceInfoToValueNode(const ValueNodePtr &value_node, std::vector<std::string> *device_formats,
                               std::vector<TypeId> *device_types) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(device_formats);
  MS_EXCEPTION_IF_NULL(device_types);
  ValuePtr value = value_node->value();
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(value, &tensors);
  if (!tensors.empty()) {
    if (tensors.size() != AnfAlgo::GetOutputTensorNum(value_node)) {
      MS_LOG(EXCEPTION) << "The size of tensors converted from value [" << tensors.size()
                        << "] is not equal to output size of value node [" << AnfAlgo::GetOutputTensorNum(value_node)
                        << "]";
    }
    device_formats->clear();
    device_types->clear();
    for (const auto &tensor : tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      auto device_sync = tensor->device_address();
      if (device_sync != nullptr) {
        auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
        MS_EXCEPTION_IF_NULL(device_address);
        device_formats->emplace_back(device_address->format());
        device_types->emplace_back(device_address->type_id());
        continue;
      }
      device_formats->emplace_back(kOpFormat_DEFAULT);
      device_types->emplace_back(kTypeUnknown);
    }
  }
}

std::string GetNodeGroup(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    return AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  }
  return "";
}

bool NeedOptimizeCommOp(const AnfNodePtr &node, std::map<std::string, std::string> *optimized_comm_group) {
  MS_EXCEPTION_IF_NULL(optimized_comm_group);
  auto node_group = GetNodeGroup(node);
  if (node_group.find(kSyncBnGroup) != string::npos) {
    return false;
  }
  auto node_name = AnfAlgo::GetCNodeName(node);
  auto iter = optimized_comm_group->find(node_name);
  if (iter == optimized_comm_group->end()) {
    (*optimized_comm_group)[node_name] = node_group;
    return true;
  } else if (iter->second == node_group) {
    return true;
  }
  return false;
}
}  // namespace

AnfNodePtr KernelGraph::MakeValueNode(const AnfNodePtr &node) {
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  ValueNodePtr new_value_node = std::make_shared<ValueNode>(value_node->value());
  new_value_node->set_abstract(value_node->abstract());
  this->SetKernelInfoForNode(new_value_node);
  return new_value_node;
}

std::vector<AnfNodePtr> KernelGraph::outputs() const {
  auto graph_output = output();
  if (IsPrimitiveCNode(graph_output, prim::kPrimMakeTuple)) {
    auto make_tuple = output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    auto &inputs = make_tuple->inputs();
    return std::vector<AnfNodePtr>(inputs.begin() + 1, inputs.end());
  }
  return std::vector<AnfNodePtr>(1, graph_output);
}

void KernelGraph::VisitNodeDescendants(const AnfNodePtr &node, std::queue<AnfNodePtr> *visit_queue,
                                       std::unordered_set<AnfNodePtr> *visited_nodes, bool comm_first) {
  MS_EXCEPTION_IF_NULL(visit_queue);
  MS_EXCEPTION_IF_NULL(visited_nodes);
  auto it = node_output_edges_.find(node);
  if (it == node_output_edges_.end()) {
    // value node and parameter has no input,no need to print log
    if (node->isa<CNode>()) {
      MS_LOG(DEBUG) << "Can not find node [" << node->DebugString() << "]";
    }
    return;
  }

  // visit all reduce node first, then other nodes
  std::vector<AnfNodePtr> active_nodes;
  for (const auto &output_edge : it->second) {
    auto next_node = output_edge.first;
    MS_EXCEPTION_IF_NULL(next_node);
    if (node_input_num_.find(next_node) == node_input_num_.end()) {
      MS_LOG(EXCEPTION) << "Can't find node[" << next_node->DebugString() << "]";
    }
    MS_LOG(DEBUG) << "Decrease input:" << next_node->DebugString() << ",node:" << node->DebugString()
                  << ",num: " << node_input_num_[next_node] << ",decrease num:" << output_edge.second;
    if (node_input_num_[next_node] < output_edge.second) {
      MS_LOG(DEBUG) << "Input node:" << next_node->DebugString() << ",node_output_num" << node_input_num_[next_node]
                    << ",depend edge:" << output_edge.second;
      continue;
    }
    node_input_num_[next_node] = node_input_num_[next_node] - output_edge.second;
    // allreduce first
    if (node_input_num_[next_node] == 0 && visited_nodes->find(next_node) == visited_nodes->end()) {
      (void)visited_nodes->insert(next_node);
      bool is_comm_node = AnfAlgo::IsCommunicationOp(next_node);
      if ((is_comm_node && comm_first) || (!is_comm_node && !comm_first)) {
        MS_LOG(DEBUG) << "Visit node:" << next_node->DebugString();
        visit_queue->push(next_node);
      } else {
        active_nodes.emplace_back(next_node);
      }
    }
  }

  for (auto &node : active_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Visit node:" << node->DebugString();
    visit_queue->push(node);
  }
}

void KernelGraph::SetExecOrderByDefault() {
  std::queue<AnfNodePtr> seed_nodes;
  UpdateNodeEdgeList(&seed_nodes);
  execution_order_.clear();
  std::unordered_set<AnfNodePtr> visited_nodes;
  std::queue<AnfNodePtr> zero_input_nodes;
  std::stack<AnfNodePtr> delay_comm_stack;
  std::queue<AnfNodePtr> communication_descendants;
  std::map<std::string, std::string> optimized_comm_group;
  while (!seed_nodes.empty() || !delay_comm_stack.empty()) {
    // seed nodes first, then delay comm nodes
    if (seed_nodes.empty()) {
      VisitNodeDescendants(delay_comm_stack.top(), &communication_descendants, &visited_nodes, false);
      delay_comm_stack.pop();
    } else {
      zero_input_nodes.push(seed_nodes.front());
      seed_nodes.pop();
    }
    // comm descendant first, then common queue
    while (!zero_input_nodes.empty() || !communication_descendants.empty()) {
      AnfNodePtr node = nullptr;
      bool is_communication_descendant = false;
      if (communication_descendants.empty()) {
        node = zero_input_nodes.front();
        zero_input_nodes.pop();
      } else {
        node = communication_descendants.front();
        communication_descendants.pop();
        is_communication_descendant = true;
      }
      // add execute node
      MS_EXCEPTION_IF_NULL(node);
      if (node->isa<CNode>() && AnfAlgo::IsRealKernel(node)) {
        execution_order_.push_back(node->cast<CNodePtr>());
      }
      // delay execute comm ops that need optimize
      bool is_fused_comm = AnfAlgo::IsFusedCommunicationOp(node);
      bool optimize_comm = is_fused_comm;
      if (optimize_comm) {
        optimize_comm = NeedOptimizeCommOp(node, &optimized_comm_group);
      }
      if (optimize_comm) {
        while (!delay_comm_stack.empty()) {
          VisitNodeDescendants(delay_comm_stack.top(), &communication_descendants, &visited_nodes, false);
          delay_comm_stack.pop();
        }
        delay_comm_stack.push(node);
      } else if (is_fused_comm) {
        delay_comm_stack.push(node);
      } else if (is_communication_descendant) {
        VisitNodeDescendants(node, &communication_descendants, &visited_nodes);
      } else {
        VisitNodeDescendants(node, &zero_input_nodes, &visited_nodes);
      }
    }
  }
  CheckLoop();
  // resort start label / end goto
  execution_order_ = SortStartLabelAndEndGoto();
}

std::vector<CNodePtr> KernelGraph::SortStartLabelAndEndGoto() {
  std::vector<CNodePtr> re_order;
  if (start_label_ != nullptr) {
    re_order.push_back(start_label_);
  }
  for (auto &node : execution_order_) {
    if (node == start_label_ || node == end_goto_) {
      continue;
    }

    if (IsSameLabel(node, end_goto_)) {
      end_goto_ = node;
      MS_LOG(INFO) << "Replace end_goto_ in kernel graph:" << graph_id();
      continue;
    }

    if (IsSameLabel(node, start_label_)) {
      start_label_ = node;
      MS_LOG(INFO) << "Replace start_label_ in kernel graph:" << graph_id();
      continue;
    }

    //
    // Re-order:
    //   u = LabelGoto(...)
    //   x = Mul(...)
    //   LabelSet(u)
    // To:
    //   u = LabelGoto(...)
    //   LabelSet(u)
    //   x = Mul(...)
    // This prevent Mul be skipped.
    //
    if (IsPrimitiveCNode(node, prim::kPrimLabelSet) && (re_order.back() != node->input(1))) {
      auto iter = std::find(re_order.rbegin() + 1, re_order.rend(), node->input(1));
      if (iter != re_order.rend()) {
        re_order.insert(iter.base(), node);
        continue;
      }
    }

    re_order.push_back(node);
  }
  if (end_goto_ != nullptr) {
    re_order.push_back(end_goto_);
  }
  return re_order;
}

void KernelGraph::GetLoopNodesByDFS(AnfNodePtr node, uint32_t *loop_num) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_input_it = node_input_edges_.find(node);
  if (node_input_it == node_input_edges_.end()) {
    MS_LOG(DEBUG) << "Node [" << node->DebugString() << "] don't have input edges.";
    return;
  }
  if (*loop_num != 0) {
    return;
  }
  visited_nodes_.insert(node);
  for (auto input_edge : node_input_edges_[node]) {
    size_t input_num = node_input_num_[input_edge.first];
    if (input_num == 0) {
      continue;
    }
    if (find(visited_nodes_.begin(), visited_nodes_.end(), input_edge.first) == visited_nodes_.end()) {
      MS_EXCEPTION_IF_NULL(input_edge.first);
      edge_to_[input_edge.first] = node;
      GetLoopNodesByDFS(input_edge.first, loop_num);
    } else {
      AnfNodePtr node_iter = node;
      MS_EXCEPTION_IF_NULL(node_iter);
      MS_LOG(INFO) << "Print loop nodes start:";
      for (; node_iter != input_edge.first && node_iter != nullptr; node_iter = edge_to_[node_iter]) {
        loop_nodes_.push(node_iter);
        node_input_num_[node_iter]--;
        MS_LOG(INFO) << "Get loop node:" << node_iter->DebugString();
      }
      if (node_iter != nullptr) {
        loop_nodes_.push(node_iter);
        loop_nodes_.push(node);
        (*loop_num)++;
        node_input_num_[node_iter]--;
        MS_LOG(INFO) << "Get loop node:" << node_iter->DebugString();
        MS_LOG(INFO) << "Get loop node:" << node->DebugString();
        MS_LOG(INFO) << "Print loop nodes end, Loop num:" << *loop_num;
        while (!loop_nodes_.empty()) {
          loop_nodes_.pop();
        }
        return;
      }
    }
  }
}

uint32_t KernelGraph::GetLoopNum(std::map<AnfNodePtr, size_t> none_zero_nodes) {
  uint32_t loop_num = 0;
  for (auto iter = none_zero_nodes.begin(); iter != none_zero_nodes.end(); iter++) {
    auto node = iter->first;
    MS_EXCEPTION_IF_NULL(node);
    if (node_input_num_[node] == 0) {
      continue;
    }
    edge_to_.clear();
    visited_nodes_.clear();
    GetLoopNodesByDFS(node, &loop_num);
  }
  return loop_num;
}

void KernelGraph::CheckLoop() {
  std::map<AnfNodePtr, size_t> none_zero_nodes;
  if (node_input_edges_.size() != node_input_num_.size()) {
    MS_LOG(EXCEPTION) << "node_input_edges_ size :" << node_input_edges_.size()
                      << "not equal to node_input_num_ size:" << node_input_num_.size();
  }
  for (auto &it : node_input_num_) {
    MS_EXCEPTION_IF_NULL(it.first);
    string str;
    auto node_input_it = node_input_edges_.find(it.first);
    if (node_input_it == node_input_edges_.end()) {
      MS_LOG(EXCEPTION) << "Can't find node [" << it.first->DebugString() << "]";
    }
    if (it.second != 0) {
      for (const auto &input_edge : node_input_edges_[it.first]) {
        MS_EXCEPTION_IF_NULL(input_edge.first);
        str = str.append(input_edge.first->DebugString()).append("|");
      }
      MS_LOG(WARNING) << "Node:" << it.first->DebugString() << ",inputs:" << str << ",input num:" << it.second;
      none_zero_nodes[it.first] = it.second;
    }
  }
  // if don't consider control depend and loop exit,a exception will be throw
  if (!none_zero_nodes.empty()) {
    MS_LOG(WARNING) << "Nums of loop:" << GetLoopNum(none_zero_nodes);
    MS_LOG(EXCEPTION) << "Nodes have loop, left node num:" << none_zero_nodes.size();
  }
}

CNodePtr KernelGraph::NewCNode(const std::vector<AnfNodePtr> &inputs) {
  auto cnode = FuncGraph::NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  cnode->set_abstract(std::make_shared<abstract::AbstractNone>());
  CreateKernelInfoFromNewParameter(cnode);
  if (AnfAlgo::GetCNodeName(cnode) == prim::kPrimCast->name()) {
    AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(false), cnode);
  }
  SetKernelInfoForNode(cnode);
  AnfAlgo::SetGraphId(graph_id_, cnode.get());
  return cnode;
}

void KernelGraph::CreateKernelInfoFromNewParameter(const CNodePtr &cnode) {
  if (!AnfAlgo::IsGraphKernel(cnode)) {
    return;
  }
  auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> node_list;
  std::vector<AnfNodePtr> input_list;
  std::vector<AnfNodePtr> output_list;
  kernel::GetValidKernelNodes(func_graph, &node_list, &input_list, &output_list);
  for (auto &anf_node : node_list) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (anf_node->kernel_info() == nullptr) {
      anf_node->set_kernel_info(std::make_shared<device::KernelInfo>());
    }
    auto anf_cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(anf_cnode);
    size_t input_num = AnfAlgo::GetInputTensorNum(anf_cnode);
    for (size_t i = 0; i < input_num; ++i) {
      auto input_node = anf_cnode->input(i + 1);
      MS_EXCEPTION_IF_NULL(input_node);
      if (IsValueNode<tensor::Tensor>(input_node)) {
        auto new_input_node = MakeValueNode(input_node);
        if (new_input_node != nullptr) {
          anf_cnode->set_input(i + 1, new_input_node);
        }
      }
    }
  }
  for (auto &anf_node : input_list) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (anf_node->kernel_info() == nullptr) {
      anf_node->set_kernel_info(std::make_shared<device::KernelInfo>());
    }
  }
}

void KernelGraph::ResetAssignInputFeaatureMapFlag(const CNodePtr &cnode) const {
  if (kOpAssignKernelNameList.find(AnfAlgo::GetCNodeName(cnode)) == kOpAssignKernelNameList.end()) {
    MS_LOG(EXCEPTION) << "Only supported to change the node [Assign , AssignSub, AssignAdd] node's input feature map "
                         "flag but got the node :"
                      << cnode->DebugString();
  }
  auto input_node = AnfAlgo::GetInputNode(cnode, 0);
  auto assign_value_node = AnfAlgo::GetInputNode(cnode, 1);
  if (AnfAlgo::IsFeatureMapOutput(input_node)) {
    return;
  }
  if (!AnfAlgo::IsFeatureMapOutput(input_node) && AnfAlgo::IsFeatureMapOutput(assign_value_node)) {
    auto kernel_info = static_cast<device::KernelInfo *>(input_node->kernel_info());
    kernel_info->set_feature_map_flag(true);
  }
}

void KernelGraph::SetKernelInfoForNode(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  node->set_kernel_info(kernel_info);
  if (node->isa<CNode>()) {
    if (kOpAssignKernelNameList.find(AnfAlgo::GetCNodeName(node)) != kOpAssignKernelNameList.end()) {
      ResetAssignInputFeaatureMapFlag(node->cast<CNodePtr>());
    }
#if defined(__APPLE__)
    std::vector<int> feature_map_input_indexs;
#else
    std::vector<size_t> feature_map_input_indexs;
#endif
    kernel_info->set_feature_map_flag(false);
    size_t input_num = AnfAlgo::GetInputTensorNum(node);
    for (size_t index = 0; index < input_num; ++index) {
      if (AnfAlgo::IsFeatureMapInput(node, index)) {
        kernel_info->set_feature_map_flag(true);
        feature_map_input_indexs.push_back(index);
      }
    }
    if (AnfAlgo::GetInputTensorNum(node) == 0) {
      kernel_info->set_feature_map_flag(true);
    }
    if (AnfAlgo::IsRealKernel(node)) {
      // if the node only has the primitive(such as getNext) or the node's input has a feature map input
      // then the node's output is a feature map output
      AnfAlgo::SetNodeAttr(kIsFeatureMapOutput, MakeValue(kernel_info->is_feature_map()), node);
      AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), node);
    }
    return;
  }
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  // set the format of value_node to DEFAULT_FORMAT
  std::vector<TypeId> types;
  std::vector<std::string> formats = {kOpFormat_DEFAULT};
  if (node->isa<ValueNode>()) {
    kernel_info->set_feature_map_flag(false);
    types.emplace_back(kTypeUnknown);
    auto value_node = node->cast<ValueNodePtr>();
    SyncDeviceInfoToValueNode(value_node, &formats, &types);
  }
  if (node->isa<Parameter>()) {
    auto parameter = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    bool is_weight = AnfAlgo::IsParameterWeight(parameter);
    kernel_info->set_feature_map_flag(!is_weight);
    types.push_back(is_weight ? kTypeUnknown : AnfAlgo::GetOutputInferDataType(parameter, 0));
  }
  // set parameter initaial device data type
  kernel_build_info_builder->SetOutputsFormat(formats);
  kernel_build_info_builder->SetOutputsDeviceType(types);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), node.get());
}

CNodePtr KernelGraph::NewCNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_cnode = std::make_shared<CNode>(*cnode);
  // if a cnode is created not from front,this cnode won't be in map,so when replace it,we shouldn't update map
  if (BackendNodeExistInFrontBackendMap(cnode)) {
    FrontBackendlMapUpdate(cnode, new_cnode);
  }
  AnfAlgo::SetGraphId(graph_id_, cnode.get());
  return new_cnode;
}

ParameterPtr KernelGraph::NewParameter(const ParameterPtr &parameter) {
  auto abstract = parameter == nullptr ? std::make_shared<abstract::AbstractNone>() : parameter->abstract();
  auto new_parameter = NewParameter(abstract);
  MS_EXCEPTION_IF_NULL(new_parameter);

  // if don't use default parameter = nullptr,it remarks create a new parameter from a old parameter
  if (parameter != nullptr) {
    new_parameter->set_name(parameter->name());
    if (AnfAlgo::IsParameterWeight(parameter)) {
      new_parameter->set_default_param(parameter->default_param());
    }
  }
  // create kernel_info form new parameter
  SetKernelInfoForNode(new_parameter);
  AnfAlgo::SetGraphId(graph_id_, new_parameter.get());
  return new_parameter;
}

ParameterPtr KernelGraph::NewParameter(const abstract::AbstractBasePtr &abstract) {
  ParameterPtr new_parameter = add_parameter();
  new_parameter->set_abstract(abstract);
  MS_EXCEPTION_IF_NULL(new_parameter);
  // create kernel_info form new parameter
  SetKernelInfoForNode(new_parameter);
  AnfAlgo::SetGraphId(graph_id_, new_parameter.get());
  return new_parameter;
}

ValueNodePtr KernelGraph::NewValueNode(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  auto new_value_node = MakeValueNode(value_node)->cast<ValueNodePtr>();
  AnfAlgo::SetGraphId(graph_id_, new_value_node.get());
  return new_value_node;
}

ValueNodePtr KernelGraph::NewValueNode(const AbstractBasePtr &abstract, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(value);
  ValueNodePtr new_value_node = std::make_shared<ValueNode>(value);
  new_value_node->set_abstract(abstract);
  SetKernelInfoForNode(new_value_node);
  AnfAlgo::SetGraphId(graph_id(), new_value_node.get());
  return new_value_node;
}

AnfNodePtr KernelGraph::TransValueNodeTuple(const AbstractBasePtr abstract, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(value);
  if (!abstract->isa<abstract::AbstractTuple>()) {
    auto new_value_node = NewValueNode(abstract, value);
    AddValueNodeToGraph(new_value_node);
    return new_value_node;
  }
  auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
  auto value_tuple = value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  MS_EXCEPTION_IF_NULL(value_tuple);
  if (tuple_abstract->size() != value_tuple->size()) {
    MS_LOG(EXCEPTION) << "Abstract size:" << tuple_abstract->size()
                      << " is not equal to value size:" << value_tuple->size();
  }
  std::vector<AnfNodePtr> make_tuple_inputs = {
    mindspore::NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  for (size_t index = 0; index < tuple_abstract->size(); ++index) {
    make_tuple_inputs.push_back(TransValueNodeTuple((*tuple_abstract)[index], (*value_tuple)[index]));
  }
  auto make_tuple = NewCNode(make_tuple_inputs);
  make_tuple->set_abstract(tuple_abstract);
  return make_tuple;
}

AnfNodePtr KernelGraph::TransParameterTuple(const AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractTuple>()) {
    return NewParameter(abstract);
  }
  auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  std::vector<AnfNodePtr> make_tuple_inputs = {
    mindspore::NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  for (size_t index = 0; index < tuple_abstract->size(); ++index) {
    make_tuple_inputs.push_back(TransParameterTuple((*tuple_abstract)[index]));
  }
  auto make_tuple = NewCNode(make_tuple_inputs);
  make_tuple->set_abstract(tuple_abstract);
  return make_tuple;
}

AnfNodePtr KernelGraph::CreatTupleGetItemNode(const AnfNodePtr &node, size_t output_idx) {
  auto idx = mindspore::NewValueNode(SizeToLong(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToLong(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  AnfNodePtr tuple_getitem = NewCNode({mindspore::NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  tuple_getitem->set_scope(node->scope());
  std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(node, output_idx);
  TypeId origin_type = AnfAlgo::GetOutputInferDataType(node, output_idx);
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, tuple_getitem.get());
  return tuple_getitem;
}

AnfNodePtr KernelGraph::TransCNodeTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<TypeId> types;
  std::vector<std::vector<size_t>> shapes;
  std::vector<AnfNodePtr> make_tuple_inputs_list = {mindspore::NewValueNode(prim::kPrimMakeTuple)};
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t tuple_out_index = 0; tuple_out_index < output_num; ++tuple_out_index) {
    make_tuple_inputs_list.emplace_back(CreatTupleGetItemNode(node, tuple_out_index));
    types.push_back(AnfAlgo::GetOutputInferDataType(node, tuple_out_index));
    shapes.emplace_back(AnfAlgo::GetOutputInferShape(node, tuple_out_index));
  }
  auto make_tuple = NewCNode(make_tuple_inputs_list);
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, make_tuple.get());
  return make_tuple;
}

AnfNodePtr KernelGraph::TransTupleToMakeTuple(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsTupleOutput(node)) {
    return node;
  }
  if (node->isa<Parameter>()) {
    return TransParameterTuple(node->abstract());
  } else if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto make_tuple = TransValueNodeTuple(value_node->abstract(), value_node->value());
    if (RemoveValueNodeFromGraph(value_node)) {
      MS_LOG(WARNING) << "Failed to remove the value_node " << value_node->DebugString();
    }
    return make_tuple;
  } else if (node->isa<CNode>()) {
    return TransCNodeTuple(node->cast<CNodePtr>());
  }
  MS_LOG(EXCEPTION) << "Unexpected node:" << node->DebugString();
}

const std::vector<AnfNodePtr> &KernelGraph::inputs() const {
  MS_EXCEPTION_IF_NULL(inputs_);
  return *inputs_;
}

void KernelGraph::FrontBackendlMapAdd(const AnfNodePtr &front_anf, const AnfNodePtr &backend_anf) {
  MS_EXCEPTION_IF_NULL(front_anf);
  MS_EXCEPTION_IF_NULL(backend_anf);
  if (front_backend_anf_map_.find(front_anf) != front_backend_anf_map_.end()) {
    MS_LOG(EXCEPTION) << "Anf " << front_anf->DebugString() << " has been exist in the front_backend_anf_map_";
  }
  if (backend_front_anf_map_.find(backend_anf) != backend_front_anf_map_.end()) {
    auto front_node = front_anf->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(front_node);
    auto attr_input = front_node->input(kAnfPrimitiveIndex);
    if (!attr_input->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Kernel " << backend_anf->DebugString() << "has been exist in the backend_front_anf_map_";
    }
  }
  front_backend_anf_map_[front_anf] = backend_anf;
  backend_front_anf_map_[backend_anf] = front_anf;
}

void KernelGraph::FrontBackendlMapUpdate(const AnfNodePtr &old_backend_anf, const AnfNodePtr &new_backend_anf) {
  MS_EXCEPTION_IF_NULL(old_backend_anf);
  MS_EXCEPTION_IF_NULL(new_backend_anf);
  if (old_backend_anf == new_backend_anf) {
    MS_LOG(DEBUG) << "Old same with new:" << old_backend_anf->DebugString();
    return;
  }
  if (backend_front_anf_map_.find(old_backend_anf) == backend_front_anf_map_.end()) {
    MS_LOG(DEBUG) << "Old_backend_anf " << old_backend_anf->DebugString() << " is not exist in the map";
    return;
  }
  if (front_backend_anf_map_.find(backend_front_anf_map_[old_backend_anf]) == front_backend_anf_map_.end()) {
    MS_LOG(EXCEPTION) << "Anf is not exist in the map ,old " << old_backend_anf->DebugString();
  }
  if (IsInternalOutput(old_backend_anf)) {
    ReplaceInternalOutput(old_backend_anf, new_backend_anf);
  }
  front_backend_anf_map_[backend_front_anf_map_[old_backend_anf]] = new_backend_anf;
  backend_front_anf_map_[new_backend_anf] = backend_front_anf_map_[old_backend_anf];
  // delete old kernel
  (void)backend_front_anf_map_.erase(old_backend_anf);
}

// get kernel by anf
AnfNodePtr KernelGraph::GetBackendAnfByFrontAnf(const AnfNodePtr &front_anf) {
  if (front_backend_anf_map_.find(front_anf) == front_backend_anf_map_.end()) {
    return nullptr;
  }
  return front_backend_anf_map_[front_anf];
}

bool KernelGraph::BackendNodeExistInFrontBackendMap(const AnfNodePtr &backend_anf) {
  return backend_front_anf_map_.find(backend_anf) != backend_front_anf_map_.end();
}

ValueNodePtr KernelGraph::GetValueNodeByTensor(const mindspore::tensor::TensorPtr &tensor) {
  if (tensor_to_value_node_map_.find(tensor) == tensor_to_value_node_map_.end()) {
    return nullptr;
  }
  return tensor_to_value_node_map_[tensor];
}

void KernelGraph::TensorValueNodeMapAdd(const tensor::TensorPtr &tensor, const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  tensor_to_value_node_map_[tensor] = value_node;
}

void KernelGraph::AddDependEdge(const AnfNodePtr &node, const AnfNodePtr &input, size_t depend_edge_num) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input);
  MS_LOG(DEBUG) << "Input:" << input->DebugString() << ",  node:" << node->DebugString() << ",num:" << depend_edge_num;
  auto output_depend_edge = std::pair<AnfNodePtr, size_t>(node, depend_edge_num);
  // add output depend edge of input
  auto output_it = node_output_edges_.find(input);
  if (output_it == node_output_edges_.end()) {
    node_output_edges_[input] = std::vector<std::pair<AnfNodePtr, size_t>>{output_depend_edge};
  } else {
    output_it->second.push_back(output_depend_edge);
  }
  // add input depend edge of output
  auto input_depend_edge = std::pair<AnfNodePtr, size_t>(input, depend_edge_num);
  auto input_it = node_input_edges_.find(node);
  if (input_it == node_input_edges_.end()) {
    node_input_edges_[node] = std::vector<std::pair<AnfNodePtr, size_t>>{input_depend_edge};
  } else {
    input_it->second.push_back(input_depend_edge);
  }
  // add node input depend num
  auto depend_it = node_input_num_.find(node);
  if (depend_it == node_input_num_.end()) {
    node_input_num_[node] = depend_edge_num;
  } else {
    depend_it->second += depend_edge_num;
  }
}

std::vector<AnfNodePtr> KernelGraph::GetOutputNodes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto it = node_output_edges_.find(node);
  if (it == node_output_edges_.end()) {
    MS_LOG(EXCEPTION) << "Can't find node[" << node->DebugString() << "]";
  }
  std::vector<AnfNodePtr> output_nodes;
  auto trans = [](const std::pair<AnfNodePtr, size_t> &pair) -> AnfNodePtr { return pair.first; };
  (void)std::transform(it->second.begin(), it->second.end(), std::back_inserter(output_nodes), trans);
  return output_nodes;
}

// update the depend relations of control depend
void KernelGraph::UpdateControlDependRelations(const std::vector<AnfNodePtr> &depends) {
  for (const auto &node : depends) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      return;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimControlDepend)) {
      MS_LOG(EXCEPTION) << node->DebugString() << " is not a control depend";
    }
    auto prior_node = cnode->input(kControlDependPriorIndex);
    auto depend_node = cnode->input(kControlDependBehindIndex);
    MS_EXCEPTION_IF_NULL(prior_node);
    MS_EXCEPTION_IF_NULL(depend_node);
    std::vector<AnfNodePtr> prior_nodes = {prior_node};
    std::vector<AnfNodePtr> depend_nodes = {depend_node};
    int depend_mode = 0;
    if (AnfAlgo::HasNodeAttr(kControlDependMode, cnode)) {
      depend_mode = AnfAlgo::GetNodeAttr<int64_t>(cnode, kControlDependMode);
    }
    MS_LOG(DEBUG) << "Prior node[" << prior_node->DebugString() << "], depend node[" << depend_node->DebugString()
                  << "], depend_mode :" << depend_mode << ".";
    if (prior_node->isa<Parameter>() && depend_mode == 1) {
      prior_nodes = GetOutputNodes(prior_node);
    }
    if (depend_node->isa<Parameter>()) {
      depend_nodes = depend_mode == 1 ? GetOutputNodes(depend_node) : std::vector<AnfNodePtr>{};
    }

    std::vector<AnfNodePtr> real_prior_nodes;
    std::set<AnfNodePtr> prior_visited;
    for (const auto &tmp : prior_nodes) {
      AnfAlgo::GetAllFatherRealNode(tmp, &real_prior_nodes, &prior_visited);
    }
    std::vector<AnfNodePtr> real_depend_nodes;
    std::set<AnfNodePtr> depend_visited;
    for (const auto &tmp : depend_nodes) {
      AnfAlgo::GetAllFatherRealNode(tmp, &real_depend_nodes, &depend_visited);
    }
    UpdateNodeInputOutputEdges(real_prior_nodes, real_depend_nodes);
  }
}

void KernelGraph::UpdateNodeInputOutputEdges(const std::vector<AnfNodePtr> &real_prior_nodes,
                                             const std::vector<AnfNodePtr> &real_depend_nodes) {
  for (auto &first_node : real_prior_nodes) {
    if (AnfAlgo::CheckPrimitiveType(first_node, prim::kPrimControlDepend)) {
      continue;
    }
    for (auto &second_node : real_depend_nodes) {
      if (AnfAlgo::CheckPrimitiveType(second_node, prim::kPrimControlDepend)) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(first_node);
      MS_EXCEPTION_IF_NULL(second_node);
      MS_LOG(DEBUG) << "Add first node:" << first_node->DebugString() << ",second node:" << second_node->DebugString();
      AddDependEdge(second_node, first_node, 1);
    }
  }
}

bool KernelGraph::HandleControlDependNode(const AnfNodePtr &node, std::queue<AnfNodePtr> *que,
                                          std::unordered_set<AnfNodePtr> *visited_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(que);
  MS_EXCEPTION_IF_NULL(visited_nodes);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimControlDepend)) {
    return false;
  }
  // set the control depend visited but don't push it into the que
  if (visited_nodes->find(node) != visited_nodes->end()) {
    return true;
  }
  (void)visited_nodes->insert(cnode);
  // add a 0 depend num to keep the link relations to prepare for finding zero output nodes
  auto prior_node = cnode->input(kControlDependPriorIndex);
  auto depend_node = cnode->input(kControlDependBehindIndex);
  for (const auto &input : cnode->inputs()) {
    AddDependEdge(node, input, 0);
  }
  PushNoVisitedNode(depend_node, que, visited_nodes);
  PushNoVisitedNode(prior_node, que, visited_nodes);
  return true;
}

void KernelGraph::UpdateNodeEdgeList(std::queue<AnfNodePtr> *seed_nodes) {
  MS_EXCEPTION_IF_NULL(seed_nodes);
  node_output_edges_.clear();
  node_input_num_.clear();
  node_input_edges_.clear();
  std::unordered_set<AnfNodePtr> visited_nodes;
  std::queue<AnfNodePtr> que;
  que.push(get_return());
  while (!que.empty()) {
    auto node = que.front();
    que.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>() || node->isa<ValueNode>()) {
      seed_nodes->push(node);
      continue;
    }
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr) {
      continue;
    }
    auto &inputs = cnode->inputs();
    // We push inputs from right to left, so that them can be evaluated from left to right.
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      auto &input = *iter;
      PushNoVisitedNode(input, &que, &visited_nodes);
      AddDependEdge(node, input, 1);
    }
  }
}

void KernelGraph::AddValueNodeToGraph(const ValueNodePtr &value_node) { (void)graph_value_nodes_.insert(value_node); }

bool KernelGraph::IsInRefOutputMap(const AnfWithOutIndex &pair) const { return ref_out_in_map_.count(pair) != 0; }

AnfWithOutIndex KernelGraph::GetRefCorrespondOutput(const AnfWithOutIndex &out_pair) const {
  if (!IsInRefOutputMap(out_pair)) {
    MS_LOG(EXCEPTION) << "Out_pair is not in RefOutputMap";
  }
  return ref_out_in_map_.at(out_pair);
}

void KernelGraph::AddRefCorrespondPairs(const AnfWithOutIndex &final_pair, const AnfWithOutIndex &origin_pair) {
  if (IsInRefOutputMap(final_pair)) {
    MS_LOG(EXCEPTION) << "Out_pair is already in RefOutputMap";
  }
  (void)ref_out_in_map_.insert(std::make_pair(final_pair, origin_pair));
}

bool KernelGraph::RemoveValueNodeFromGraph(const ValueNodePtr &value_node) {
  if (graph_value_nodes_.find(value_node) != graph_value_nodes_.end()) {
    (void)graph_value_nodes_.erase(value_node);
    return true;
  }
  return false;
}

void KernelGraph::ReplaceGraphInput(const AnfNodePtr &old_parameter, const AnfNodePtr &new_parameter) {
  // update graph inputs
  MS_EXCEPTION_IF_NULL(old_parameter);
  MS_EXCEPTION_IF_NULL(new_parameter);
  if (old_parameter == new_parameter) {
    return;
  }
  for (size_t i = 0; i < inputs_->size(); i++) {
    if ((*inputs_)[i] == old_parameter) {
      MS_LOG(INFO) << "Replace input of graph:" << graph_id_ << ", old graph input: " << old_parameter->DebugString()
                   << ",new graph input:" << new_parameter->DebugString();
      (*inputs_)[i] = new_parameter;
      break;
    }
  }
}

void KernelGraph::ReplaceNode(NotNull<AnfNodePtr> old_anf_node, NotNull<AnfNodePtr> new_anf_node) {
  MS_EXCEPTION_IF_NULL(inputs_);
  {
    std::queue<AnfNodePtr> seed_nodes;
    UpdateNodeEdgeList(&seed_nodes);
  }
  auto it = node_output_edges_.find(old_anf_node);
  if (it != node_output_edges_.end()) {
    const auto &outputs = it->second;
    for (auto &output_node : outputs) {
      MS_EXCEPTION_IF_NULL(output_node.first);
      auto output_cnode = output_node.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      auto &output_node_inputs = output_cnode->inputs();
      // don't replace node if it is a control edge  => output_node.second == 0
      if (output_node.second == 0) {
        continue;
      }
      for (size_t i = 1; i < output_node_inputs.size(); i++) {
        if (output_node_inputs[i] == old_anf_node.get()) {
          output_cnode->set_input(i, new_anf_node);
        }
      }
    }
    // update front to backend map
    FrontBackendlMapUpdate(old_anf_node, new_anf_node);
  }
  {
    std::queue<AnfNodePtr> seed_nodes;
    UpdateNodeEdgeList(&seed_nodes);
  }
}

void KernelGraph::UpdateExecuteKernelStreamLabel() {
  for (auto &kernel : execution_order_) {
    AnfAlgo::SetStreamDistinctionLabel(stream_distinction_label_, kernel.get());
  }
}

std::vector<std::shared_ptr<KernelGraph>> KernelGraph::GetLeafGraphOrder() {
  std::vector<std::shared_ptr<KernelGraph>> leaf_graph_order;
  if (IsLeafGraph()) {
    leaf_graph_order.push_back(shared_from_this()->cast<KernelGraphPtr>());
  } else {
    for (const auto &child_graph : child_graph_order_) {
      std::shared_ptr<KernelGraph> child_graph_ptr = child_graph.lock();
      MS_EXCEPTION_IF_NULL(child_graph_ptr);
      auto child_leaf_graph_order = child_graph_ptr->GetLeafGraphOrder();
      std::copy(child_leaf_graph_order.begin(), child_leaf_graph_order.end(), std::back_inserter(leaf_graph_order));
    }
  }
  return leaf_graph_order;
}

bool KernelGraph::IsLeafGraph() const { return child_graph_order_.empty(); }

std::vector<CNodePtr> KernelGraph::FindNodeByPrimitive(const PrimitivePtr &primitive) const {
  std::vector<CNodePtr> result;
  for (const auto &anf : execution_order_) {
    if (AnfAlgo::CheckPrimitiveType(anf, primitive) && AnfAlgo::GetGraphId(anf.get()) == graph_id_) {
      result.push_back(anf->cast<CNodePtr>());
    }
  }
  return result;
}

std::vector<CNodePtr> KernelGraph::FindNodeByPrimitive(const std::vector<PrimitivePtr> &primitive_list) const {
  std::vector<CNodePtr> result;
  for (const auto &anf : execution_order_) {
    for (const auto &primitive : primitive_list) {
      if (AnfAlgo::CheckPrimitiveType(anf, primitive) && AnfAlgo::GetGraphId(anf.get()) == graph_id_) {
        result.push_back(anf->cast<CNodePtr>());
      }
    }
  }
  return result;
}

void KernelGraph::PrintGraphExecuteOrder() const {
  if (!(IS_OUTPUT_ON(INFO))) {
    return;
  }
  MS_LOG(INFO) << "Graph " << graph_id_ << " execution order:";
  for (size_t i = 0; i < execution_order_.size(); i++) {
    CNodePtr cur_cnode_ptr = execution_order_[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);

    std::string event_str;
    if (AnfAlgo::HasNodeAttr(kAttrEventId, cur_cnode_ptr)) {
      event_str = ", event id[" + std::to_string(AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrEventId)) + "]";
    }

    std::string label_str;
    if (AnfAlgo::HasNodeAttr(kAttrLabelIndex, cur_cnode_ptr)) {
      label_str = ", label id[" + std::to_string(AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrLabelIndex)) + "]";
    }

    if (AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, cur_cnode_ptr)) {
      auto label_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cur_cnode_ptr, kAttrLabelSwitchList);
      label_str = ", label id[";
      for (size_t j = 0; j < label_list.size(); ++j) {
        label_str += std::to_string(label_list[j]) + (j + 1 < label_list.size() ? ", " : "]");
      }
    }

    std::string active_stream_str;
    if (AnfAlgo::HasNodeAttr(kAttrActiveStreamList, cur_cnode_ptr)) {
      auto stream_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cur_cnode_ptr, kAttrActiveStreamList);
      active_stream_str = ", active stream id[";
      for (size_t j = 0; j < stream_list.size(); ++j) {
        active_stream_str += std::to_string(stream_list[j]) + (j + 1 < stream_list.size() ? ", " : "]");
      }
    }

    std::string group_str;
    if (AnfAlgo::GetKernelType(cur_cnode_ptr) == HCCL_KERNEL && AnfAlgo::HasNodeAttr(kAttrGroup, cur_cnode_ptr)) {
      group_str = ", group[" + AnfAlgo::GetNodeAttr<std::string>(cur_cnode_ptr, kAttrGroup) + "]";
    }

    MS_LOG(INFO) << "Index[" << i << "], node name[" << cur_cnode_ptr->fullname_with_scope() << "], logic id["
                 << AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get()) << "], stream id["
                 << AnfAlgo::GetStreamId(cur_cnode_ptr) << "], node info[" << cur_cnode_ptr->DebugString() << "]"
                 << event_str << label_str << active_stream_str << group_str;
  }
}

void KernelGraph::AddInternalOutput(const AnfNodePtr &front_node, const AnfNodePtr &node, int output_idx,
                                    bool unique_target) {
  if (front_node == nullptr || node == nullptr) {
    MS_LOG(INFO) << "Front node or node is nullptr";
    return;
  }
  MS_LOG(INFO) << "Add internal node " << node->DebugString() << " with front node " << front_node->DebugString();
  front_to_internal_outputs_map_[front_node] = node;
  if (AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimTupleGetItem)) {
    output_idx = AnfAlgo::GetTupleGetItemOutIndex(front_node->cast<CNodePtr>());
  }
  internal_outputs_to_front_map_[node][output_idx] = std::pair<AnfNodePtr, bool>(front_node, unique_target);
}

void KernelGraph::AddInternalOutputTensor(const AnfNodePtr &node, int output_idx, const tensor::TensorPtr &tensor) {
  if (node == nullptr) {
    return;
  }
  internal_outputs_tensor_map_[node][output_idx] = tensor;
}

tensor::TensorPtr KernelGraph::GetInternalOutputTensor(const AnfNodePtr &node, int output_idx) {
  if (node == nullptr) {
    return nullptr;
  }
  auto iter = internal_outputs_tensor_map_.find(node);
  if (iter == internal_outputs_tensor_map_.end()) {
    return nullptr;
  }
  auto idx_iter = iter->second.find(output_idx);
  if (idx_iter == iter->second.end()) {
    return nullptr;
  }
  return idx_iter->second;
}

void KernelGraph::ReplaceInternalOutput(const AnfNodePtr &node, const AnfNodePtr &new_node, int src_output_idx,
                                        int dst_output_idx) {
  if (new_node == nullptr || node == nullptr) {
    MS_LOG(INFO) << "New node or node is nullptr";
    return;
  }
  if (node == new_node) {
    MS_LOG(INFO) << "New node and node is the same";
    return;
  }
  auto iter = internal_outputs_to_front_map_.find(node);
  if (iter == internal_outputs_to_front_map_.end()) {
    MS_LOG(INFO) << "Node is not internal output";
    return;
  }
  MS_LOG(INFO) << "Replace internal node " << node->DebugString() << " To " << new_node->DebugString();
  auto &front_nodes = iter->second;
  // Move all front nodes to new node mapping
  if (src_output_idx == -1) {
    internal_outputs_to_front_map_[new_node] = front_nodes;
    for (const auto &front_node_iter : front_nodes) {
      front_to_internal_outputs_map_[front_node_iter.second.first] = new_node;
    }
    internal_outputs_to_front_map_.erase(iter);
    return;
  }
  // Move specified front node to new node mapping
  auto front_node_iter = front_nodes.find(src_output_idx);
  if (front_node_iter == front_nodes.end()) {
    MS_LOG(INFO) << "The output " << src_output_idx << " of node " << node->DebugString() << " is not an internal node";
    return;
  }
  auto front_node_pair = front_node_iter->second;
  internal_outputs_to_front_map_[new_node][dst_output_idx] = front_node_pair;
  front_to_internal_outputs_map_[front_node_pair.first] = new_node;
  front_nodes.erase(src_output_idx);
  if (front_nodes.empty()) {
    internal_outputs_to_front_map_.erase(iter);
  }
}

AnfNodePtr KernelGraph::GetInternalOutputByFrontNode(const AnfNodePtr &front_node) const {
  auto iter = front_to_internal_outputs_map_.find(front_node);
  if (iter != front_to_internal_outputs_map_.end()) {
    return iter->second;
  }
  return nullptr;
}

bool KernelGraph::IsInternalOutput(const AnfNodePtr &node, int output_idx) const {
  auto front_nodes_iter = internal_outputs_to_front_map_.find(node);
  if (front_nodes_iter == internal_outputs_to_front_map_.end()) {
    return false;
  }
  if (output_idx == -1) {
    return true;
  }
  auto &front_nodes = front_nodes_iter->second;
  if (front_nodes.find(output_idx) == front_nodes.end()) {
    return false;
  }
  return true;
}

bool KernelGraph::IsUniqueTargetInternalOutput(const AnfNodePtr &node, int output_idx) const {
  auto front_nodes_iter = internal_outputs_to_front_map_.find(node);
  if (front_nodes_iter == internal_outputs_to_front_map_.end()) {
    return false;
  }
  auto &front_nodes = front_nodes_iter->second;
  auto idx_iter = front_nodes.find(output_idx);
  if (idx_iter == front_nodes.end()) {
    return false;
  }
  return idx_iter->second.second;
}

void KernelGraph::UpdateChildGraphOrder() {
  MS_LOG(INFO) << "Update " << ToString() << " child graph order.";
  SetExecOrderByDefault();
  auto call_nodes = FindNodeByPrimitive({std::make_shared<Primitive>(prim::kPrimCall->name()),
                                         std::make_shared<Primitive>(prim::kPrimSwitch->name()),
                                         std::make_shared<Primitive>(prim::kPrimSwitchLayer->name())});
  std::vector<std::weak_ptr<KernelGraph>> child_graph_order;
  for (auto &call_node : call_nodes) {
    MS_EXCEPTION_IF_NULL(call_node);
    auto call_child_graphs = AnfAlgo::GetCallSwitchKernelGraph(call_node->cast<CNodePtr>());
    for (const auto &child_graph : call_child_graphs) {
      MS_EXCEPTION_IF_NULL(child_graph);
      if (child_graph != parent_graph_.lock()) {
        auto shared_this = std::dynamic_pointer_cast<KernelGraph>(shared_from_this());
        MS_EXCEPTION_IF_NULL(shared_this);
        child_graph->set_parent_graph(shared_this);
      }
      child_graph_order.push_back(child_graph);
    }
  }
  for (size_t i = 0; i < child_graph_order.size(); ++i) {
    std::shared_ptr<KernelGraph> child_graph = child_graph_order[i].lock();
    MS_EXCEPTION_IF_NULL(child_graph);
    MS_LOG(INFO) << "Child graph[" << i << "][id:" << child_graph->graph_id() << "]";
  }
  child_graph_order_ = child_graph_order;
}

void KernelGraph::RemoveNodeFromGraph(const AnfNodePtr &node) {
  if (backend_front_anf_map_.find(node) != backend_front_anf_map_.end()) {
    auto front_node = backend_front_anf_map_[node];
    (void)backend_front_anf_map_.erase(node);
    (void)front_backend_anf_map_.erase(front_node);
  }
  if (node->isa<ValueNode>()) {
    if (graph_value_nodes_.find(node->cast<ValueNodePtr>()) != graph_value_nodes_.end()) {
      (void)graph_value_nodes_.erase(node->cast<ValueNodePtr>());
    }
  }
}

ParameterPtr KernelGraph::AddExtraParamAndTensor(std::string param_name, int32_t value) {
  ParameterPtr param;
  ShapeVector shp = {1};
  tensor::TensorPtr tensor_ptr = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  mindspore::abstract::AbstractBasePtr paremeter_abstract_ptr = tensor_ptr->ToAbstract();
  ParameterPtr new_param = std::make_shared<Parameter>(shared_from_this()->cast<KernelGraphPtr>());
  MS_EXCEPTION_IF_NULL(new_param);
  new_param->set_name(param_name);
  new_param->set_abstract(paremeter_abstract_ptr);
  param = NewParameter(new_param);
  // ensure alloc mem for this param
  std::vector<AnfNodePtr> *mute_inputs = MutableInputs();
  MS_EXCEPTION_IF_NULL(mute_inputs);
  mute_inputs->push_back(param);

  tensor::TensorPtr data_tensor_ptr = std::make_shared<tensor::Tensor>(kInt32->type_id(), shp);
  MS_EXCEPTION_IF_NULL(data_tensor_ptr);
  int32_t *val = nullptr;
  val = static_cast<int32_t *>(data_tensor_ptr->data_c());
  *val = value;

  extra_param_tensor_.push_back(std::make_pair(param, data_tensor_ptr));
  MS_LOG(INFO) << "Create new param: " << param->DebugString();
  return param;
}

void KernelGraph::UpdateGraphDynamicAttr() {
  for (const auto &cnode : execution_order_) {
    if (AnfAlgo::IsDynamicShape(cnode)) {
      MS_LOG(INFO) << "Update Graph Dynamic Attr";
      is_dynamic_shape_ = true;
      return;
    }
  }
  is_dynamic_shape_ = false;
}

void KernelGraph::SetInputNodes() {
  input_nodes_.clear();
  for (const auto &input_node : inputs()) {
    auto params = AnfAlgo::GetAllOutput(input_node);
    std::copy(params.begin(), params.end(), std::back_inserter(input_nodes_));
  }
}

void KernelGraph::SetOptimizerFlag() {
  has_optimizer_ = false;
  for (const auto &cnode : execution_order_) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto node_name = AnfAlgo::GetCNodeName(cnode);
    if (kOptOperatorSet.find(node_name) != kOptOperatorSet.end()) {
      has_optimizer_ = true;
      return;
    }
    if (node_name.find("Assign") != string::npos) {
      for (auto &input : cnode->inputs()) {
        MS_EXCEPTION_IF_NULL(input);
        if (input->isa<Parameter>() && AnfAlgo::IsParameterWeight(input->cast<ParameterPtr>())) {
          has_optimizer_ = true;
          return;
        }
      }
    }
  }
}

std::string KernelGraph::ToString() const { return std::string("kernel_graph_").append(std::to_string(graph_id_)); }

KernelGraph::~KernelGraph() {
  device::KernelRuntimeManager::Instance().ClearGraphResource(graph_id_, *inputs_, graph_value_nodes_,
                                                              execution_order_);
}
}  // namespace session
}  // namespace mindspore
