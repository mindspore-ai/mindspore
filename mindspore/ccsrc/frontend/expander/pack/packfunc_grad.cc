/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "frontend/expander/pack/packfunc_grad.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include <map>
#include "frontend/expander/pack/pack_expander.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/pynative/pynative_utils.h"
#include "include/backend/optimizer/helper.h"
#include "ops/conv_pool_op_name.h"
#include "ops/math_op_name.h"

namespace mindspore {
namespace expander {
namespace {
using grad_common = pynative::PyNativeAlgo::GradCommon;

void GetUsedForwardNodesInBprop(const FuncGraphPtr &func_graph, AnfNodePtrList *node_list) {
  const auto &order = TopoSort(func_graph->output());
  mindspore::HashSet<AnfNodePtr> batch_norm_nodes;
  for (const auto &node : order) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      MS_LOG(EXCEPTION) << "Should be primitive, but: " << node->DebugString();
    }
    if (!grad_common::IsRealOp(node)) {
      continue;
    }
    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString();
    auto op_name = prim->name();
    auto unused_inputs = BpropExpander::GetUnusedInputs(op_name);
    grad_common::GetUsedCNodeInBpropGraph(cnode, unused_inputs, node_list);
  }
  // remove same node
  mindspore::OrderedSet node_list_set(*node_list);
  node_list->assign(node_list_set.begin(), node_list_set.end());
}

static std::vector<AnfNodePtr> GetNodesFromTuple(const FuncGraphPtr &graph, const AnfNodePtr &tuple_output,
                                                 const size_t output_size) {
  std::vector<AnfNodePtr> nodes;
  for (size_t index = 0; index < output_size; ++index) {
    nodes.push_back(opt::CreatTupleGetItemNode(graph, tuple_output, index));
  }
  return nodes;
}

// Modify the output node of func_graph to add forward nodes used in bprop graph.
static void ModifyOutputNode(const FuncGraphPtr &func_graph, const GraphGradInfoPtr &graph_grad_info) {
  MS_LOG_DEBUG << "start ModifyOutputNode";
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->used_forward_nodes().empty()) {
    return;
  }
  // Get original output node
  auto original_output_node = func_graph->output();
  MS_EXCEPTION_IF_NULL(original_output_node);
  std::vector<AnfNodePtr> output_nodes;
  if (IsPrimitiveCNode(original_output_node, prim::kPrimMakeTuple)) {
    auto &inputs = original_output_node->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      output_nodes.push_back(inputs[i]);
    }
  } else {
    output_nodes.push_back(original_output_node);
  }
  auto original_output_size = output_nodes.size();
  // save node to index
  std::map<common::KernelWithIndex, size_t> node2index;
  for (size_t i = 0; i < output_nodes.size(); ++i) {
    const auto &node = output_nodes[i];
    const auto &kernel_index = common::AnfAlgo::VisitKernel(node, 0);
    node2index.insert(std::make_pair(kernel_index, i));
  }
  // Get used forward nodes
  size_t output_index = output_nodes.size();
  abstract::AbstractBasePtrList added_abs_list;
  std::vector<size_t> forward_node_output_index;
  for (const auto &forward_node : func_graph->used_forward_nodes()) {
    MS_EXCEPTION_IF_NULL(forward_node);
    auto cnode = forward_node->cast<CNodePtr>();
    const auto &forward_vnode = cnode->forward().first;
    std::vector<AnfNodePtr> nodes;
    if (auto abs = forward_node->abstract()->cast<abstract::AbstractTuplePtr>()) {
      nodes = GetNodesFromTuple(func_graph, forward_node, abs->size());
    } else {
      nodes.push_back(forward_node);
    }
    MS_EXCEPTION_IF_NULL(graph_grad_info);
    graph_grad_info->forward_vnodes.emplace_back(forward_vnode, nodes.size());
    for (const auto &node : nodes) {
      const auto &kernel_index = common::AnfAlgo::VisitKernel(node, 0);
      const auto iter = node2index.find(kernel_index);
      if (iter != node2index.end()) {
        auto index = iter->second;
        forward_node_output_index.push_back(index);
        continue;
      }
      output_nodes.push_back(node);
      forward_node_output_index.push_back(output_index++);
    }
  }
  graph_grad_info->forward_node_output_index = forward_node_output_index;
  graph_grad_info->added_output_size = output_index - original_output_size;
  func_graph->ClearUsedForwardNodes();
  if (graph_grad_info->added_output_size > 0) {
    // merge original output node and used forward nodes to return node
    std::vector<AnfNodePtr> new_output_nodes{NewValueNode(prim::kPrimMakeTuple)};
    new_output_nodes.insert(new_output_nodes.end(), output_nodes.begin(), output_nodes.end());
    abstract::AbstractBasePtrList new_output_abs;
    std::transform(output_nodes.begin(), output_nodes.end(), std::back_inserter(new_output_abs),
                   [](const AnfNodePtr node_) { return node_->abstract(); });
    auto merge_node = func_graph->NewCNode(std::move(new_output_nodes));
    merge_node->set_abstract(std::make_shared<abstract::AbstractTuple>(new_output_abs));
    func_graph->set_output(merge_node);
    func_graph->set_modify_output(true);
  }
  MS_LOG_DEBUG << "end ModifyOutputNode";
}

static mindspore::HashMap<int64_t, GraphGradInfoPtr> graph_grad_info_cache;

void CacheGraphGradInfo(int64_t graph_id, const GraphGradInfoPtr &graph_grad_info) {
  graph_grad_info->graph_id = graph_id;
  graph_grad_info_cache[graph_grad_info->graph_id] = graph_grad_info;
}
}  // namespace

const GraphGradInfoPtr &GetGraphGradInfo(int64_t graph_id) {
  const auto iter = graph_grad_info_cache.find(graph_id);
  if (iter == graph_grad_info_cache.end()) {
    MS_LOG_EXCEPTION << "Can not get GraphGradInfo from cache for graph " << graph_id;
  }
  return iter->second;
}

void ClearGraphGradInfoCache() { graph_grad_info_cache.clear(); }

GraphGradInfoPtr GenGraphGradInfo(const FuncGraphPtr &func_graph) {
  auto graph_grad_info = std::make_shared<GraphGradInfo>();
  auto func_graph_added_output = BasicClone(func_graph);
  AnfNodePtrList node_list;
  GetUsedForwardNodesInBprop(func_graph_added_output, &node_list);
  grad_common::SetForward(node_list);
  auto func_graph_set_forward = BasicClone(func_graph_added_output);
  func_graph_added_output->set_used_forward_nodes(node_list);
  ModifyOutputNode(func_graph_added_output, graph_grad_info);
  graph_grad_info->ori_graph = func_graph;
  graph_grad_info->graph_set_forward = func_graph_set_forward;
  graph_grad_info->ori_output_abs = func_graph->output()->abstract();
  graph_grad_info->graph = func_graph_added_output;
  CacheGraphGradInfo(graph_grad_info->graph->debug_info()->get_id(), graph_grad_info);
  return graph_grad_info;
}

const mindspore::HashSet<size_t> GetUnusedInputs(const FuncGraphPtr &func_graph) {
  mindspore::HashSet<AnfNodePtr> unused_input_nodes;
  mindspore::HashMap<AnfNodePtr, size_t> node_map_index;
  auto parameters = func_graph->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto node = parameters[i];
    unused_input_nodes.insert(node);
    node_map_index[node] = i;
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &order = TopoSort(func_graph->output());
  // Hard coded and redundant with other code, this code needs to be optimized in the future
  static mindspore::HashSet<std::string> kMulOp{
    kMulOpName,
    kMatMulOpName,
    kConv2DOpName,
  };
  for (const auto &node : order) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString();
    auto op_name = AnfUtils::GetCNodeName(cnode);
    auto unused_inputs = BpropExpander::GetUnusedInputs(op_name);
    if (kMulOp.find(op_name) != kMulOp.end()) {
      // For operators like Mul, the dx ONLY rely on y, and dy ONLY rely on x.
      // so if y is a valuenode, the dy is useless, we can free x in ahead.
      if (cnode->input(kIndex1)->isa<ValueNode>()) {
        unused_inputs.insert(kIndex1);
      }
      if (cnode->input(kIndex2)->isa<ValueNode>()) {
        unused_inputs.insert(kIndex0);
      }
    }
    auto is_unused_index = [&unused_inputs](size_t i) {
      return std::find(unused_inputs.begin(), unused_inputs.end(), i) != unused_inputs.end();
    };
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto input_node = cnode->input(i);
      if (unused_input_nodes.count(input_node) > 0 && !is_unused_index(i - 1)) {
        unused_input_nodes.erase(input_node);
      }
    }
  }
  mindspore::HashSet<size_t> unused_input_index;
  for (auto node : unused_input_nodes) {
    unused_input_index.insert(node_map_index.at(node));
  }
  return unused_input_index;
}
}  // namespace expander
}  // namespace mindspore
