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

#include "plugin/device/gpu/optimizer/trt_pass/graph_partitioner.h"

#include <memory>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <utility>
#include <string>
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/optimizer/trt_pass/trt_op_factory.h"
#include "backend/graph_compiler/segment_runner.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {
bool WeightCheck(const AnfNodePtr &node) {
  static std::map<std::string, std::set<size_t>> weight_list = {
    {kConv2DOpName, {1}}, {kBatchNormOpName, {1, 2, 3, 4}}, {kConv2DBackpropInputOpName, {1}}};

  MS_EXCEPTION_IF_NULL(node);
  const std::string &op_name = common::AnfAlgo::GetCNodePrimitive(node)->name();
  auto iter = weight_list.find(op_name);
  if (iter != weight_list.end()) {
    std::vector<session::KernelWithIndex> real_inputs;
    common::AnfAlgo::GetRealInputs(node, &real_inputs);

    for (auto index : iter->second) {
      if (index >= real_inputs.size()) {
        MS_LOG(EXCEPTION) << "index out of range. node: " << node->DebugString() << ", index: " << index
                          << real_inputs.size() << trace::DumpSourceLines(node);
      }

      if (real_inputs[index].first->isa<Parameter>() &&
          !common::AnfAlgo::IsParameterWeight(real_inputs[index].first->cast<ParameterPtr>())) {
        return false;
      }
    }

    return true;
  }
  return true;
}

mindspore::HashMap<AnfNodePtr, NodeInfo> CollectNodeInfo(const FuncGraphPtr &func_graph) {
  mindspore::HashMap<AnfNodePtr, NodeInfo> res;

  const std::vector<AnfNodePtr> &node_list = TopoSort(func_graph->get_return());
  for (size_t i = 0; i < node_list.size(); i++) {
    const auto &node = node_list[i];
    if (!node->isa<CNode>()) {
      continue;
    }

    if (!AnfUtils::IsRealKernel(node)) {
      res[node] = NodeInfo(NodeType::kSupport, i);
      continue;
    }

    const std::string &op_name = common::AnfAlgo::GetCNodePrimitive(node)->name();
    const auto &converter_factory = TrtOpFactory::GetInstance();
    ConvertFunc convert_func = converter_factory.GetConvertFunc(op_name);
    if (!convert_func) {
      res[node] = NodeInfo(NodeType::kUnsupported, i);
      continue;
    }

    // Trt requires certain input to be weight.
    res[node] = WeightCheck(node) ? NodeInfo(NodeType::kSupport, i) : NodeInfo(NodeType::kUnsupported, i);
  }

  return res;
}
}  // namespace

void GraphDependency::InheritDependency(const string &lhs, const string &rhs) {
  const auto &iter = dependencies_.find(rhs);
  if (iter != dependencies_.end()) {
    dependencies_[lhs].insert(iter->second.begin(), iter->second.end());
  }
}

bool GraphDependency::ExistDependency(const string &lhs, const string &rhs) const {
  const auto &iter = dependencies_.find(lhs);
  if (iter == dependencies_.end()) {
    return false;
  }

  if (iter->second.count(rhs) == 0) {
    return false;
  }

  return true;
}

std::string GraphDependency::ToString() const {
  std::ostringstream output_buffer;
  for (const auto &graph : dependencies_) {
    output_buffer << graph.first << ": ";
    for (const auto &dependency : graph.second) {
      output_buffer << dependency << " ";
    }
    output_buffer << std::endl;
  }
  return output_buffer.str();
}

void GraphPartitioner::NewSubGraph(NodeInfo *node_info) {
  static size_t trt_id = 0;
  static size_t native_id = 0;

  node_info->graph_id_ = node_info->type() == NodeType::kSupport ? std::string("T_") + std::to_string(trt_id++)
                                                                 : std::string("N_") + std::to_string(native_id++);
}

bool GraphPartitioner::ExistCycleAfterMerge(const AnfNodePtr &node, const std::string &target_graph_id) {
  MS_EXCEPTION_IF_NULL(node);

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_node = common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), input_index);
    if (input_node == nullptr || input_node->isa<ValueNode>() || input_node->isa<Parameter>()) continue;

    if (node_info_[input_node].graph_id() == target_graph_id) {
      continue;
    }

    if (dependency_.ExistDependency(node_info_[input_node].graph_id(), target_graph_id)) {
      return true;
    }
  }
  return false;
}

void GraphPartitioner::MergeParentBranchRecursively(const AnfNodePtr &node, const std::string &old_graph_id,
                                                    const std::string &new_graph_id) {
  if (old_graph_id == new_graph_id) {
    return;
  }

  MS_EXCEPTION_IF_NULL(node);
  node_info_[node].graph_id_ = new_graph_id;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_node = common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), input_index);
    if (input_node == nullptr || input_node->isa<ValueNode>() || input_node->isa<Parameter>()) continue;

    if (node_info_[input_node].graph_id() == old_graph_id) {
      MergeParentBranchRecursively(input_node, old_graph_id, new_graph_id);
    }
  }
}

bool GraphPartitioner::NodeGrouping(const FuncGraphPtr &func_graph) {
  const std::vector<AnfNodePtr> &node_list = TopoSort(func_graph->get_return());
  for (const auto &item : node_list) {
    const auto &node = item->cast<CNodePtr>();
    if (node == nullptr) continue;

    // Alloc new subgraph temporarily. May be merged to other parent subgraph latter.
    NodeInfo &current_node_info = node_info_[node];
    NewSubGraph(&current_node_info);

    // Merge the `TupleGetItem` node too parent sub graph.
    if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      const AnfNodePtr &input = common::AnfAlgo::GetTupleGetItemRealInput(node);
      const NodeInfo &parent_node_info = node_info_[input];
      current_node_info.graph_id_ = parent_node_info.graph_id_;
      continue;
    }

    size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      auto input_node = common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), input_index);
      if (input_node == nullptr || input_node->isa<ValueNode>() || input_node->isa<Parameter>()) continue;

      const NodeInfo &parent_node_info = node_info_[input_node];
      // Alloc new graph either type is different or parent is `TupleGetItem`. For example:
      // Graph:         Split     ->  TupleGetItem  ->  Mul
      // Annotation:    Unsupported   Support           Support
      // Result:        Native        Native            Trt
      if (current_node_info.type() != parent_node_info.type() ||
          (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimTupleGetItem) &&
           current_node_info.graph_id().at(0) != parent_node_info.graph_id().at(0))) {
        dependency_.AddDependency(current_node_info.graph_id(), parent_node_info.graph_id());
        dependency_.InheritDependency(current_node_info.graph_id(), parent_node_info.graph_id());
        continue;
      }

      // Current and parent with same node type.
      // Try to merge current node to parent sub graph.

      if (ExistCycleAfterMerge(node, parent_node_info.graph_id())) {
        dependency_.InheritDependency(current_node_info.graph_id(), parent_node_info.graph_id());
      } else {
        // Merge current node to parent subgraph
        if (current_node_info.final() == false) {
          dependency_.InheritDependency(parent_node_info.graph_id_, current_node_info.graph_id_);
          current_node_info.graph_id_ = parent_node_info.graph_id_;
        } else {
          // Merge branch 2 to branch 1 if branch 1, branch 2 and current node type are same.
          //   b1   b2      b1   b1
          //    \   /  -->   \   /
          //      n            n
          // Using copy instead of reference, as `MergeParentBranchRecursively` will modify it indirectly.
          const string old_graph_id = parent_node_info.graph_id();
          const string new_graph_id = current_node_info.graph_id();
          MergeParentBranchRecursively(input_node, old_graph_id, new_graph_id);
          dependency_.InheritDependency(new_graph_id, old_graph_id);
        }

        current_node_info.final_ = true;
      }
    }
  }
  return true;
}

std::map<std::string, AnfNodePtrList> GraphPartitioner::CollectSegments() {
  std::map<std::string, AnfNodePtrList> segments;
  for (const auto &item : node_info_) {
    const std::string &graph_id = item.second.graph_id();
    if (graph_id.find("T_") != graph_id.npos &&
        common::AnfAlgo::GetCNodePrimitive(item.first)->name() != kReturnOpName) {
      segments[graph_id].push_back(item.first);
    }
  }

  for (auto &segment : segments) {
    std::sort(segment.second.begin(), segment.second.end(), [this](const AnfNodePtr &node1, const AnfNodePtr &node2) {
      size_t index1 = this->node_info_[node1].topo_index();
      size_t index2 = this->node_info_[node2].topo_index();
      return index1 < index2;
    });
  }

  return segments;
}

Subgraph GraphPartitioner::CreateNewGraph(const AnfNodePtrList &segment) {
  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(segment);
  FuncGraphManagerPtr mng = fg->manager();
  if (mng == nullptr) {
    mng = Manage(fg, false);
    fg->set_manager(mng);
  }

  // Copy inputs names and value for building Trt network.
  std::vector<AnfNodePtr> parameter_nodes = fg->parameters();
  for (size_t i = 0; i < parameter_nodes.size(); i++) {
    ParameterPtr dst = parameter_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(dst);
    if (inputs[i]->isa<Parameter>()) {
      ParameterPtr src = inputs[i]->cast<ParameterPtr>();
      dst->set_name(src->name());
      dst->debug_info()->set_name(src->name());

      if (common::AnfAlgo::IsParameterWeight(src)) {
        dst->set_default_param(src->default_param());
      }
    } else {
      dst->set_name(inputs[i]->fullname_with_scope());
      dst->debug_info()->set_name(inputs[i]->fullname_with_scope());
    }
  }

  return std::make_tuple(fg, inputs, outputs);
}

std::map<std::string, AnfNodePtrList> GraphPartitioner::Partition(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  node_info_ = CollectNodeInfo(func_graph);

  bool ret = NodeGrouping(func_graph);
  if (!ret) {
    MS_LOG(WARNING) << "Classify nodes failed.";
  }

  return CollectSegments();
}
}  // namespace opt
}  // namespace mindspore
