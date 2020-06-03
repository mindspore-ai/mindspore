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
#include "session/kernel_graph.h"
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <set>
#include "operator/ops.h"
#include "ir/param_value_py.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_info.h"
#include "kernel/kernel_build_info.h"
#include "device/kernel_runtime_manager.h"

namespace mindspore {
namespace session {
namespace {
constexpr auto kIsFeatureMapOutput = "IsFeatureMapOutput";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
void PushNoVisitedNode(const AnfNodePtr &node, std::queue<AnfNodePtr> *que,
                       std::unordered_set<AnfNodePtr> *visited_nodes) {
  MS_EXCEPTION_IF_NULL(que);
  MS_EXCEPTION_IF_NULL(visited_nodes);
  if (visited_nodes->find(node) == visited_nodes->end()) {
    que->push(node);
    (void)visited_nodes->insert(node);
    MS_LOG(DEBUG) << "Push que:" << node->DebugString();
  }
}

std::vector<AnfNodePtr> GetCallRealOutputs(const AnfNodePtr &call_node) {
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(call_node, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  if (!AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimCall)) {
    return {item_with_index.first};
  }
  std::vector<AnfNodePtr> real_inputs;
  auto child_graphs = AnfAlgo::GetCallNodeKernelGraph(item_with_index.first->cast<CNodePtr>());
  for (const auto &child_graph : child_graphs) {
    if (child_graph->get_output_null()) {
      continue;
    }
    auto real_input = child_graph->output();
    auto child_real_inputs = GetCallRealOutputs(real_input);
    std::copy(child_real_inputs.begin(), child_real_inputs.end(), std::back_inserter(real_inputs));
  }
  return real_inputs;
}
}  // namespace
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
                                       std::unordered_set<AnfNodePtr> *visited_nodes) {
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
    if (node_input_num_.find(next_node) == node_input_num_.end()) {
      MS_EXCEPTION_IF_NULL(next_node);
      MS_LOG(EXCEPTION) << "Can't find node[" << next_node->DebugString() << "]";
    }
    MS_EXCEPTION_IF_NULL(next_node);
    MS_LOG(DEBUG) << "Decrease input:" << next_node->DebugString() << ",node:" << node->DebugString()
                  << ",num: " << node_input_num_[next_node] << ",decrease num:" << output_edge.second;
    if (node_input_num_[next_node] < output_edge.second) {
      MS_LOG(EXCEPTION) << "Input node:" << next_node->DebugString() << ",node_output_num" << node_input_num_[next_node]
                        << ",depend edge:" << output_edge.second;
    }
    node_input_num_[next_node] = node_input_num_[next_node] - output_edge.second;
    // allreduce first
    if (node_input_num_[next_node] == 0 && visited_nodes->find(next_node) == visited_nodes->end()) {
      (void)visited_nodes->insert(next_node);
      if (AnfAlgo::IsCommunicationOp(next_node)) {
        MS_LOG(DEBUG) << "visit node:" << next_node->DebugString();
        visit_queue->push(next_node);
      } else {
        active_nodes.emplace_back(next_node);
      }
    }
  }

  for (auto &node : active_nodes) {
    MS_LOG(DEBUG) << "visit node:" << node->DebugString();
    visit_queue->push(node);
  }
}

void KernelGraph::SetExecOrderByDefault() {
  std::queue<AnfNodePtr> seed_nodes;
  UpdateNodeEdgeList(&seed_nodes);
  execution_order_.clear();
  std::unordered_set<AnfNodePtr> visited_nodes;
  std::queue<AnfNodePtr> zero_input_nodes;
  AnfNodePtr last_communication_node = nullptr;
  std::queue<AnfNodePtr> communication_descendants;
  while (!seed_nodes.empty() || last_communication_node != nullptr) {
    // seed nodes first, then visit last all reduce node descendant
    if (seed_nodes.empty()) {
      VisitNodeDescendants(last_communication_node, &communication_descendants, &visited_nodes);
      last_communication_node = nullptr;
    } else {
      zero_input_nodes.push(seed_nodes.front());
      seed_nodes.pop();
    }
    // all reduce node descendant first, then common queue
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
      // for all reduce node, visit last all reduce node descendant
      if (AnfAlgo::IsCommunicationOp(node)) {
        if (last_communication_node != nullptr) {
          VisitNodeDescendants(last_communication_node, &communication_descendants, &visited_nodes);
        }
        last_communication_node = node;
      } else if (is_communication_descendant) {
        VisitNodeDescendants(node, &communication_descendants, &visited_nodes);
      } else {
        VisitNodeDescendants(node, &zero_input_nodes, &visited_nodes);
      }
    }
  }
  CheckLoop();
  // resort start label / end goto
  std::vector<CNodePtr> re_order;
  if (start_label_ != nullptr) {
    re_order.push_back(start_label_);
  }
  for (auto &node : execution_order_) {
    if (node == start_label_ || node == end_goto_) {
      continue;
    }
    re_order.push_back(node);
  }
  if (end_goto_ != nullptr) {
    re_order.push_back(end_goto_);
  }
  execution_order_ = re_order;
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
    for (const auto &input_edge : node_input_edges_[it.first]) {
      MS_EXCEPTION_IF_NULL(input_edge.first);
      str = str.append(input_edge.first->DebugString()).append("|");
    }
    if (it.second != 0) {
      MS_LOG(WARNING) << "Node:" << it.first->DebugString() << ",inputs:" << str << ",input num:" << it.second;
      none_zero_nodes[it.first] = it.second;
    }
  }
  // if don't consider control depend and loop exit,a exception will be throw
  if (!none_zero_nodes.empty()) {
    MS_LOG(EXCEPTION) << "Nodes have loop, left node num:" << none_zero_nodes.size();
  }
}

CNodePtr KernelGraph::NewCNode(const std::vector<AnfNodePtr> &inputs) {
  auto cnode = FuncGraph::NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  cnode->set_abstract(std::make_shared<abstract::AbstractNone>());
  // create kernel_info from new parameter
  auto kernel_info = std::make_shared<device::KernelInfo>();
  std::vector<size_t> feature_map_input_indexs;
  // if the node only has the primitive(such as getNext) or the node's input has a feature map input
  // then the node's output is a feature map output
  for (size_t index = 1; index < inputs.size(); ++index) {
    auto node = inputs[index];
    if (AnfAlgo::IsFeatureMapOutput(node)) {
      feature_map_input_indexs.push_back(index);
    }
  }
  if (AnfAlgo::GetCNodeName(cnode) == prim::kPrimCast->name()) {
    AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(false), cnode);
  }
  if (inputs.size() == 1 || !feature_map_input_indexs.empty()) {
    kernel_info->SetFeatureMapFlag(true);
  }
  if (AnfAlgo::IsRealCNodeKernel(cnode)) {
    AnfAlgo::SetNodeAttr(kIsFeatureMapOutput, MakeValue(kernel_info->is_feature_map()), cnode);
    AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), cnode);
  }
  cnode->set_kernel_info(kernel_info);
  AnfAlgo::SetGraphId(graph_id_, cnode.get());
  return cnode;
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
  ParameterPtr new_parameter = add_parameter();
  MS_EXCEPTION_IF_NULL(new_parameter);
  // create kernel_info form new parameter
  auto kernel_info = std::make_shared<device::KernelInfo>();
  size_t output_tensor_num = 1;
  // if use default parameter = nullptr,it remarks create a new parameter from no parameter
  if (parameter == nullptr) {
    new_parameter->set_abstract(std::make_shared<abstract::AbstractNone>());
    kernel_info->SetFeatureMapFlag(true);
  } else {
    // if don't use default parameter = nullptr,it remarks create a new parameter from a old parameter
    new_parameter->set_abstract(parameter->abstract());
    new_parameter->set_name(parameter->name());
    if (AnfAlgo::IsParameterWeight(parameter)) {
      auto param_value = std::dynamic_pointer_cast<ParamValuePy>(parameter->default_param());
      auto param_value_new = std::make_shared<ParamValuePy>(param_value->value());
      new_parameter->set_default_param(param_value_new);
      kernel_info->SetFeatureMapFlag(false);
    } else {
      kernel_info->SetFeatureMapFlag(true);
    }
    // if output is a tuple tensor,now can use for loop to handle tuple tensor
    output_tensor_num = AnfAlgo::GetOutputTensorNum(parameter);
  }
  new_parameter->set_kernel_info(kernel_info);
  // create kernel_build_info for new parameter
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  // create init data type,
  std::vector<TypeId> init_data_type = {};
  for (size_t i = 0; i < output_tensor_num; i++) {
    TypeId infer_data_type = AnfAlgo::GetOutputInferDataType(new_parameter, i);
    init_data_type.push_back(AnfAlgo::IsParameterWeight(new_parameter) ? kTypeUnknown : infer_data_type);
  }
  // set the format of parameter to DEFAULT_FORMAT
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>(output_tensor_num, kOpFormat_DEFAULT));
  // set parameter initaial device data type
  kernel_build_info_builder->SetOutputsDeviceType(init_data_type);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_parameter.get());
  AnfAlgo::SetGraphId(graph_id_, new_parameter.get());
  return new_parameter;
}

std::vector<AnfNodePtr> KernelGraph::SplitTupleValueNodeToNodeList(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  auto node_value = value_node->value();
  auto output_size = AnfAlgo::GetOutputTensorNum(value_node);
  std::vector<AnfNodePtr> convert_inputs;
  if (!node_value->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "multiple output valuenode's value must be a value tuple but got " << node_value->ToString();
  }
  auto value_tuple = node_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  if (value_tuple->size() != output_size) {
    MS_LOG(EXCEPTION) << "value tuple size" << value_tuple->size()
                      << " is not mathced with the value node's output size" << output_size;
  }
  for (size_t index = 0; index < value_tuple->value().size(); ++index) {
    auto new_value_node = std::make_shared<ValueNode>(value_tuple->value()[index]);
    AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(value_node, index)},
                                        {AnfAlgo::GetOutputInferShape(value_node, index)}, new_value_node.get());
    AddValueNodeToGraph(new_value_node);
    auto kernel_info = std::make_shared<device::KernelInfo>();
    new_value_node->set_kernel_info(kernel_info);
    kernel_info->SetFeatureMapFlag(false);
    // create kernel_build_info for new value node
    auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    // set the format of value_node to DEFAULT_FORMAT
    kernel_build_info_builder->SetOutputsFormat({kOpFormat_DEFAULT});
    // set value node initial device data type = infer data type
    kernel_build_info_builder->SetOutputsDeviceType({kTypeUnknown});
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
    AnfAlgo::SetGraphId(graph_id_, new_value_node.get());
    AddValueNodeToGraph(new_value_node);
    convert_inputs.emplace_back(new_value_node);
  }
  if (!RemoveValueNodeFromGraph(value_node)) {
    MS_LOG(WARNING) << "failed to remove the value_node " << value_node->DebugString();
  }
  return convert_inputs;
}

ValueNodePtr KernelGraph::NewValueNode(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  ValueNodePtr new_value_node = std::make_shared<ValueNode>(value_node->value());
  new_value_node->set_abstract(value_node->abstract());
  // create kernel_info fo new value node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->SetFeatureMapFlag(false);
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  // set the format of value_node to DEFAULT_FORMAT
  auto output_tensor_num = AnfAlgo::GetOutputTensorNum(value_node);
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>(output_tensor_num, kOpFormat_DEFAULT));
  // set value node initial device data type = infer data type
  std::vector<TypeId> types = std::vector<TypeId>(output_tensor_num, kTypeUnknown);
  kernel_build_info_builder->SetOutputsDeviceType(types);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph_id_, new_value_node.get());
  return new_value_node;
}

const std::vector<AnfNodePtr> &KernelGraph::inputs() const {
  MS_EXCEPTION_IF_NULL(inputs_);
  return *inputs_;
}

void KernelGraph::FrontBackendlMapAdd(const AnfNodePtr &front_anf, const AnfNodePtr &backend_anf) {
  MS_EXCEPTION_IF_NULL(front_anf);
  MS_EXCEPTION_IF_NULL(backend_anf);
  if (front_backend_anf_map_.find(front_anf) != front_backend_anf_map_.end()) {
    MS_LOG(EXCEPTION) << "anf " << front_anf->DebugString() << " has been exist in the front_backend_anf_map_";
  }
  if (backend_front_anf_map_.find(backend_anf) != backend_front_anf_map_.end()) {
    MS_LOG(EXCEPTION) << "kernel " << backend_anf->DebugString() << "has been exist in the backend_front_anf_map_";
  }
  front_backend_anf_map_[front_anf] = backend_anf;
  backend_front_anf_map_[backend_anf] = front_anf;
}

void KernelGraph::FrontBackendlMapUpdate(const AnfNodePtr &old_backend_anf, const AnfNodePtr &new_backend_anf) {
  MS_EXCEPTION_IF_NULL(old_backend_anf);
  MS_EXCEPTION_IF_NULL(new_backend_anf);
  if (old_backend_anf == new_backend_anf) {
    MS_LOG(INFO) << "old:" << old_backend_anf->DebugString() << ",new:" << new_backend_anf->DebugString();
    MS_LOG(EXCEPTION) << "old can't be same with new";
  }
  if (backend_front_anf_map_.find(old_backend_anf) == backend_front_anf_map_.end()) {
    MS_LOG(DEBUG) << "old_backend_anf " << old_backend_anf->DebugString() << " is not exist in the map";
    return;
  }
  if (front_backend_anf_map_.find(backend_front_anf_map_[old_backend_anf]) == front_backend_anf_map_.end()) {
    MS_LOG(EXCEPTION) << "anf is not exist in the map ,old " << old_backend_anf->DebugString();
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
    MS_LOG(INFO) << "Prior node[" << prior_node->DebugString() << "], depend node[" << depend_node->DebugString();
    if (prior_node->isa<Parameter>()) {
      prior_nodes = GetOutputNodes(prior_node);
    }
    if (depend_node->isa<Parameter>()) {
      depend_nodes = GetOutputNodes(depend_node);
    }
    for (auto &first_node : prior_nodes) {
      if (AnfAlgo::CheckPrimitiveType(first_node, prim::kPrimControlDepend)) {
        continue;
      }
      for (auto &second_node : depend_nodes) {
        if (AnfAlgo::CheckPrimitiveType(second_node, prim::kPrimControlDepend)) {
          continue;
        }
        MS_EXCEPTION_IF_NULL(first_node);
        MS_EXCEPTION_IF_NULL(second_node);
        MS_LOG(INFO) << "Add first node:" << first_node->DebugString() << ",second node:" << second_node->DebugString();
        AddDependEdge(second_node, first_node, 1);
      }
    }
  }
}

bool KernelGraph::HandleControlDependNode(const AnfNodePtr &node, std::queue<AnfNodePtr> *que,
                                          std::unordered_set<AnfNodePtr> *visited_nodes) {
  MS_EXCEPTION_IF_NULL(node);
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
  node_output_edges_.clear();
  node_input_num_.clear();
  node_input_edges_.clear();
  std::vector<AnfNodePtr> control_depends;
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
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // handle data links
    for (const auto &input : cnode->inputs()) {
      size_t depend_edge_num = 1;
      // handle control depend,all inputs of control depend has no depend edge
      if (HandleControlDependNode(input, &que, &visited_nodes)) {
        control_depends.push_back(input);
        depend_edge_num = 0;
      }
      PushNoVisitedNode(input, &que, &visited_nodes);
      AddDependEdge(node, input, depend_edge_num);
    }
  }
  UpdateControlDependRelations(control_depends);
}

void KernelGraph::AddValueNodeToGraph(const ValueNodePtr &value_node) { (void)graph_value_nodes_.insert(value_node); }

bool KernelGraph::IsInRefOutputMap(const AnfWithOutIndex &pair) const { return ref_out_in_map_.count(pair) != 0; }

AnfWithOutIndex KernelGraph::GetRefCorrespondOutput(const AnfWithOutIndex &out_pair) const {
  if (!IsInRefOutputMap(out_pair)) {
    MS_LOG(EXCEPTION) << "out_pair is not in RefOutputMap";
  }
  return ref_out_in_map_.at(out_pair);
}

void KernelGraph::AddRefCorrespondPairs(const AnfWithOutIndex &final_pair, const AnfWithOutIndex &origin_pair) {
  if (IsInRefOutputMap(final_pair)) {
    MS_LOG(EXCEPTION) << "out_pair is already in RefOutputMap";
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

void KernelGraph::ReplaceNode(NotNull<AnfNodePtr> old_anf_node, NotNull<AnfNodePtr> new_anf_node) {
  MS_EXCEPTION_IF_NULL(inputs_);
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
      // update graph inputs
      for (size_t i = 0; i < inputs_->size(); i++) {
        if ((*inputs_)[i] == old_anf_node.get()) {
          MS_LOG(INFO) << "Replace input of graph:" << graph_id_ << ", old graph input: " << old_anf_node->DebugString()
                       << ",new graph input:" << new_anf_node->DebugString();
          (*inputs_)[i] = new_anf_node.get();
          break;
        }
      }
    }
    // update front to backend map
    FrontBackendlMapUpdate(old_anf_node, new_anf_node);
    // update output depend relations
    node_output_edges_[new_anf_node.get()] = it->second;
    (void)node_output_edges_.erase(old_anf_node);
  }
  // update graph inputs in child graph
  auto it_real_inputs = real_inputs_.find(old_anf_node);
  if (it_real_inputs != real_inputs_.end()) {
    // insert new parameter to map
    auto iter = real_inputs_.find(new_anf_node);
    if (iter != real_inputs_.end()) {
      MS_LOG(WARNING) << new_anf_node->DebugString() << " already exist in real inputs, will be rewrited.";
      iter->second = it_real_inputs->second;
    } else {
      real_inputs_[new_anf_node.get()] = it_real_inputs->second;
    }
    // erase old parameter in map
    real_inputs_.erase(old_anf_node);
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
      MS_EXCEPTION_IF_NULL(child_graph);
      auto child_leaf_graph_order = child_graph->GetLeafGraphOrder();
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

void KernelGraph::SetRealInput(const AnfNodePtr &parameter, const AnfNodePtr &arg) {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(arg);
  MS_LOG(INFO) << "parameter: " << parameter->DebugString() << ", real input : " << arg->DebugString();
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(arg);
  if (real_inputs_.find(parameter) == real_inputs_.end()) {
    real_inputs_[parameter] = std::set<AnfNodePtr>();
  }
  auto &args = real_inputs_[parameter];
  (void)args.insert(arg);
}

std::set<AnfNodePtr> KernelGraph::GetRealInput(const AnfNodePtr &parameter) {
  MS_EXCEPTION_IF_NULL(parameter);
  auto iter = real_inputs_.find(parameter);
  if (iter != real_inputs_.end()) {
    return iter->second;
  }
  MS_LOG(EXCEPTION) << parameter->DebugString() << " not found.";
}

void KernelGraph::UpdateCallRealInput() {
  MS_LOG(INFO) << "Update graph id: " << graph_id_;
  std::map<AnfNodePtr, std::set<AnfNodePtr>> real_inputs_map;
  for (auto &it : real_inputs_) {
    auto parameter = it.first;
    MS_EXCEPTION_IF_NULL(parameter);
    auto real_inputs = it.second;
    std::vector<AnfNodePtr> new_real_inputs;
    std::set<AnfNodePtr> erase_real_inputs;
    for (auto &real_input : real_inputs) {
      // if real input is a call node ,find the child graph output act as the new real input
      auto item_with_index = AnfAlgo::VisitKernelWithReturnType(real_input, 0);
      MS_EXCEPTION_IF_NULL(item_with_index.first);
      if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimCall)) {
        (void)erase_real_inputs.insert(item_with_index.first);
        new_real_inputs = GetCallRealOutputs(item_with_index.first);
        continue;
      }
    }
    for (auto &erase_node : erase_real_inputs) {
      MS_LOG(INFO) << "paramter: " << parameter->DebugString() << " erase real input:" << erase_node->DebugString();
      (void)real_inputs.erase(erase_node);
    }
    for (auto &new_real_input : new_real_inputs) {
      MS_LOG(INFO) << "paramter: " << parameter->DebugString()
                   << " insert real input:" << new_real_input->DebugString();
      (void)real_inputs.insert(new_real_input);
    }
    real_inputs_map[parameter] = real_inputs;
  }
  real_inputs_ = real_inputs_map;
}

void KernelGraph::PrintGraphExecuteOrder() const {
  MS_LOG(INFO) << "graph:" << graph_id_ << "execution order";
  for (size_t i = 0; i < execution_order_.size(); i++) {
    CNodePtr cur_cnode_ptr = execution_order_[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kSendOpName || AnfAlgo::GetCNodeName(cur_cnode_ptr) == kRecvOpName) {
      auto primitive = AnfAlgo::GetCNodePrimitive(cur_cnode_ptr);
      MS_LOG(INFO) << "index[" << i << "], node name[" << AnfAlgo::GetCNodeName(cur_cnode_ptr) << "], logic id["
                   << AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get()) << "], stream id["
                   << AnfAlgo::GetStreamId(cur_cnode_ptr) << "], event_id["
                   << GetValue<uint32_t>(primitive->GetAttr(kAttrEventId)) << "], node info["
                   << cur_cnode_ptr->DebugString() << "]";
    } else {
      MS_LOG(INFO) << "index[" << i << "], node name[" << cur_cnode_ptr->fullname_with_scope() << "], logic id["
                   << AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get()) << "], stream id["
                   << AnfAlgo::GetStreamId(cur_cnode_ptr) << "], node info[" << cur_cnode_ptr->DebugString() << "]";
    }
  }
}

std::string KernelGraph::ToString() const { return std::string("kernel_graph_").append(std::to_string(graph_id_)); }

KernelGraph::~KernelGraph() { device::KernelRuntimeManager::Instance().ClearGraphResource(graph_id_); }
}  // namespace session
}  // namespace mindspore
