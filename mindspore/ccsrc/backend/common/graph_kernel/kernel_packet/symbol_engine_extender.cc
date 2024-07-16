/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/kernel_packet/symbol_engine_extender.h"

#include <algorithm>
#include <memory>
#include <functional>
#include <vector>
#include "utils/anf_utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "mindspore/core/symbolic_shape/operation_builder.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "backend/common/graph_kernel/kernel_packet/kernel_packet_engine.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/pass/insert_type_transform_op.h"

namespace mindspore::graphkernel::packet {
using symshape::DependOn;

inline bool IsHostOp(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  if (AnfAlgo::IsKernelSelectBackoffOp(node)) {
    return true;
  }
  // ops inserted in InsertTypeTransformOp
  return opt::IsBackOffOp(node->cast<CNodePtr>());
}

inline bool IsDeviceOp(const AnfNodePtr &node) {
  if (!AnfUtils::IsRealKernel(node) || IsHostOp(node) || node->kernel_info() == nullptr) {
    return false;
  }
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  if (build_info != nullptr && build_info->valid()) {
    return true;
  }
  return false;
}

bool SymbolEngineExtender::CheckBaseNode(const AnfNodePtr &node) {
  if (GetCNodePrimitive(node) == nullptr) {
    return false;
  }
  if (!IsDeviceOp(node)) {
    return false;
  }
  auto &flags = GraphKernelFlags::GetInstance();
  if (!flags.enable_packet_ops_only.empty()) {
    if (std::find(flags.enable_packet_ops_only.begin(), flags.enable_packet_ops_only.end(),
                  AnfUtils::GetCNodeName(node)) == flags.enable_packet_ops_only.end()) {
      return false;
    }
  } else if (std::find(flags.disable_packet_ops.begin(), flags.disable_packet_ops.end(),
                       AnfUtils::GetCNodeName(node)) != flags.disable_packet_ops.end()) {
    return false;
  }
  MS_LOG(DEBUG) << "Search from the base node: " << node->DebugString();
  return true;
}

void SymbolEngineExtender::FindShapeDependHostNode(const CNodePtr &node, HashSet<AnfNodePtr> *visited,
                                                   HashSet<AnfNodePtr> *valid_nodes) {
  if (!visited->insert(node).second) {
    return;
  }
  auto prim = GetCNodePrimitive(node);
  if (prim == nullptr) {
    return;
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return;
  }
  auto depends = symshape::GetShapeDepends(prim, node->size() - 1);
  if (depends.empty()) {
    MS_LOG(DEBUG) << "The node " << node->fullname_with_scope() << " shape depend status is empty.";
    return;
  }
  MS_LOG(DEBUG) << "Add " << node->fullname_with_scope() << " into candidates.";
  (void)valid_nodes->insert(node);
  for (size_t i = 0; i < depends.size(); i++) {
    auto inp = node->input(i + 1)->cast<CNodePtr>();
    if (inp == nullptr) {
      continue;
    }
    // assume that building shape for host op does not depend input value again.
    if (depends[i] == DependOn::kShape && IsHostOp(inp)) {
      FindShapeDependHostNode(inp, visited, valid_nodes);
    }
  }
}

void SymbolEngineExtender::FindValueDependNode(const CNodePtr &node, HashSet<AnfNodePtr> *visited,
                                               HashSet<AnfNodePtr> *valid_nodes) {
  if (!visited->insert(node).second) {
    return;
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return;
  }
  auto prim = GetCNodePrimitive(node);
  if (prim == nullptr) {
    return;
  }
  auto depends = symshape::GetValueDepends(prim, node->size() - 1);
  // always try to fuse host op, if the node does not support symbolic value, the whole packet will be dropped.
  // only fuse device op when it supports building symbolic value.
  if (depends.empty() && !IsHostOp(node)) {
    MS_LOG(DEBUG) << "The " << node->fullname_with_scope() << " is not host op and value depend status is empty.";
    return;
  }
  MS_LOG(DEBUG) << "Add " << node->fullname_with_scope() << " into candidates.";
  (void)valid_nodes->insert(node);
  for (size_t i = 0; i < depends.size(); i++) {
    auto inp = node->input(i + 1)->cast<CNodePtr>();
    if (inp == nullptr) {
      continue;
    }
    if (depends[i] == DependOn::kValue) {
      FindValueDependNode(inp, visited, valid_nodes);
    } else if (IsHostOp(inp)) {
      MS_LOG(DEBUG) << "The input[" << i << "] is host op.";
      FindShapeDependHostNode(inp, visited, valid_nodes);
    }
  }
}

AnfNodePtrList SymbolEngineExtender::FindCandidates(const CNodePtr &base_node) {
  HashSet<AnfNodePtr> visited;
  HashSet<AnfNodePtr> valid_nodes;
  auto depends = symshape::GetShapeDepends(GetCNodePrimitive(base_node), base_node->size() - 1);
  if (depends.empty()) {
    return {};
  }
  // use dfs to find the clusterable nodes.
  for (size_t i = 0; i < depends.size(); i++) {
    auto inp = base_node->input(i + 1);
    if (!inp->isa<CNode>()) {
      continue;
    }
    if (depends[i] == DependOn::kValue) {
      MS_LOG(DEBUG) << "The input[" << i << "] " << inp->fullname_with_scope() << " is value-depended.";
      FindValueDependNode(inp->cast<CNodePtr>(), &visited, &valid_nodes);
    } else if (IsHostOp(inp)) {
      MS_LOG(DEBUG) << "The input[" << i << "] " << inp->fullname_with_scope()
                    << " is not value-depended, but it's a host op.";
      FindValueDependNode(inp->cast<CNodePtr>(), &visited, &valid_nodes);
    }
  }
  if (valid_nodes.empty()) {
    return {};
  }
  (void)valid_nodes.insert(base_node);

  return TopoSort(base_node, SuccIncoming, [&valid_nodes](const AnfNodePtr &node) -> IncludeType {
    return valid_nodes.count(node) > 0 ? FOLLOW : EXCLUDE;
  });
}

ValuePtr SymbolEngineExtender::FindOnlyDependShapeInputs(const FuncGraphPtr &fg) const {
  const auto &params = fg->parameters();
  std::vector<bool> only_depend_shape(params.size(), true);
  auto engine = fg->symbol_engine();
  MS_EXCEPTION_IF_NULL(engine);
  // depend value when infer
  for (size_t i = 0; i < params.size(); i++) {
    if (engine->IsDependValue(params[i])) {
      only_depend_shape[i] = false;
    }
  }
  // depend value when launch kernel
  auto kernel = fg->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(kernel);
  for (auto inp : kernel->inputs()) {
    auto iter = std::find(params.begin(), params.end(), inp);
    if (iter != params.end()) {
      only_depend_shape[iter - params.begin()] = false;
    }
  }
  return MakeValue<std::vector<bool>>(only_depend_shape);
}

CNodePtr CreatePacketNode(const FuncGraphPtr &main_fg, const FuncGraphPtr &sub_fg, const AnfNodePtrList &inputs) {
  std::vector<AnfNodePtr> fn_inputs;
  fn_inputs.reserve(inputs.size() + 1);
  (void)fn_inputs.emplace_back(NewValueNode(sub_fg));
  (void)fn_inputs.insert(fn_inputs.end(), inputs.cbegin(), inputs.cend());
  auto new_cnode = main_fg->NewCNode(fn_inputs);
  new_cnode->set_abstract(sub_fg->output()->abstract());
  new_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  return new_cnode;
}

bool SymbolEngineExtender::ExtendNode(const AnfNodePtr &node, const FuncGraphPtr &main_fg) {
  ClusterConfig config;
  config.inline_sub_func_graph = false;
  config.sort_parameter = true;

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto nodes = FindCandidates(cnode);
  if (nodes.size() <= 1) {
    return false;
  }
  MS_LOG(DEBUG) << "The size of list of nodes to be clustered: " << nodes.size();
  config.only_output_basenode = node;
  // Check if the symbol engine supports inferring for the graph, if not, skip cluster of this graph
  auto [fg, inputs, outputs] = BuildSingleGraphFromNodes(nodes, config);
  if (outputs.size() != 1) {
    MS_LOG(DEBUG) << "The size of outputs should be 1, but got " << outputs.size();
    return false;
  }
  auto symbol_engine = KernelPacketEngine::Build(fg);
  if (!symbol_engine->SupportInfer()) {
    MS_LOG(DEBUG) << "Symbol engine doesn't support infer shape from node: " << node->fullname_with_scope();
    return false;
  }
  auto new_cnode = CreatePacketNode(main_fg, fg, inputs);
  if (!common::AnfAlgo::IsDynamicShape(new_cnode)) {
    MS_LOG(DEBUG) << "The node " << new_cnode->DebugString() << " is not dynamic shape";
    return false;
  }
  auto fuse_op_name = GkUtils::ExtractGraphKernelName(nodes, "", "extend");
  fg->set_attr(kAttrKernelPacketNode, MakeValue(fuse_op_name));
  fg->set_attr("only_depend_shape", FindOnlyDependShapeInputs(fg));
  new_cnode->AddAttr(kAttrToPrim, MakeValue(AnfUtils::GetCNodeName(node) + "_packet"));
  MS_LOG(INFO) << "Replace " << node->fullname_with_scope() << " with " << new_cnode->fullname_with_scope();
  (void)main_fg->manager()->Replace(node, new_cnode);
  return true;
}

bool SymbolEngineExtender::Run(const FuncGraphPtr &func_graph) {
  // Find the manager for the FuncGraph.
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  // Find all cnodes.
  auto cnodes = TopoSort(func_graph->output(), SuccIncoming, [](const AnfNodePtr &node) {
    if (node->isa<CNode>()) {
      return FOLLOW;
    }
    return EXCLUDE;
  });

  bool changed = false;
  // Process each subgraph.
  for (auto cnode : cnodes) {
    if (!CheckBaseNode(cnode)) {
      continue;
    }
    if (ExtendNode(cnode, func_graph)) {
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel::packet
