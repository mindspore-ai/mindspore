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

#include "backend/ms_backend/graph_fusion/kernel_packet/symbol_engine_extender.h"

#include <algorithm>
#include <memory>
#include <functional>
#include <utility>
#include "utils/anf_utils.h"
#include "primitive/framework_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/array_ops.h"
#include "primitive/arithmetic_ops.h"
#include "abstract/symbolic_shape/operation_builder.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "ir/graph_utils.h"
#include "backend/ms_backend/graph_fusion/core/graph_builder.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_engine.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "ir/func_graph_flag.h"

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

bool IsOnlyOneUser(const AnfNodePtr &node) {
  auto mng = node->func_graph()->manager();
  if (mng == nullptr) {
    return false;
  }
  return mng->node_users()[node].size() == 1;
}

std::vector<DependOn> GetInferShapeDepend(const CNodePtr &node) {
  if (common::AnfAlgo::HasNodeAttr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, node) &&
      common::AnfAlgo::HasNodeAttr(kAttrFuncGraph, node)) {
    return std::vector<DependOn>(node->size() - 1, DependOn::kShape);
  }
  return symshape::GetShapeDepends(GetCNodePrimitive(node), node->size() - 1);
}

std::vector<DependOn> GetInferValueDepend(const CNodePtr &node) {
  return symshape::GetValueDepends(GetCNodePrimitive(node), node->size() - 1);
}

bool SymbolEngineExtender::CheckBaseNode(const AnfNodePtr &node) const {
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

bool SymbolEngineExtender::IsValidNode(const CNodePtr &node) const {
  if (GetCNodePrimitive(node) == nullptr) {
    return false;
  }
  if (AnfUtils::IsRealKernel(node)) {
    return true;
  }
  return IsPrimitiveCNode(node, prim::kPrimTupleGetItem);
}

void SymbolEngineExtender::SearchInputs(const CNodePtr &node, const std::vector<DependOn> &depends, size_t depth) {
  for (size_t i = 0; i < depends.size(); i++) {
    auto inp = node->input(i + 1)->cast<CNodePtr>();
    if (inp == nullptr) {
      continue;
    }
    if (depends[i] == DependOn::kValue) {
      MS_LOG(DEBUG) << "Depth-" << depth << ": The input[" << i << "] (" << node->fullname_with_scope()
                    << ") is value-depended op.";
      FindValueDependNode(inp, depth + 1);
    } else if (IsHostOp(inp)) {
      MS_LOG(DEBUG) << "Depth-" << depth << ": The input[" << i << "] (" << node->fullname_with_scope()
                    << ") is shape-depended host op.";
      FindShapeDependNode(inp, depth + 1);
    } else if (IsOnlyOneUser(inp)) {
      MS_LOG(DEBUG) << "Depth-" << depth << ": The input[" << i << "] (" << node->fullname_with_scope()
                    << ") is only-shape-depended device op.";
      FindShapeDependNode(inp, depth + 1);
    }
  }
}

void SymbolEngineExtender::FindShapeDependNode(const CNodePtr &node, size_t depth) {
  if (!visited_.insert(node).second) {
    return;
  }
  if (!IsValidNode(node)) {
    return;
  }
  auto depends = GetInferShapeDepend(node);
  if (depends.empty()) {
    MS_LOG(DEBUG) << "The node " << node->fullname_with_scope() << " shape depend status is empty.";
    return;
  }
  MS_LOG(DEBUG) << "Depth-" << depth << ": Add " << node->fullname_with_scope() << " into candidates.";
  (void)valid_nodes_.insert(node);
  SearchInputs(node, depends, depth);
}

void SymbolEngineExtender::FindValueDependNode(const CNodePtr &node, size_t depth) {
  if (!visited_.insert(node).second) {
    return;
  }
  if (!IsValidNode(node)) {
    return;
  }
  auto depends = GetInferValueDepend(node);
  if (depends.empty()) {
    MS_LOG(DEBUG) << "The " << node->fullname_with_scope() << " value depend status is empty.";
    return;
  }
  MS_LOG(DEBUG) << "Depth-" << depth << ": Add " << node->fullname_with_scope() << " into candidates.";
  (void)valid_nodes_.insert(node);
  SearchInputs(node, depends, depth);
}

void SymbolEngineExtender::RemoveWildGetitem() {
  for (auto iter = valid_nodes_.begin(); iter != valid_nodes_.end();) {
    if (IsPrimitiveCNode(*iter, prim::kPrimTupleGetItem)) {
      if (valid_nodes_.count((*iter)->cast<CNodePtr>()->input(1)) == 0) {
        iter = valid_nodes_.erase(iter);
        continue;
      }
    }
    ++iter;
  }
}

AnfNodePtrList SymbolEngineExtender::FindCandidates(const CNodePtr &base_node) {
  auto depends = symshape::GetShapeDepends(GetCNodePrimitive(base_node), base_node->size() - 1);
  if (depends.empty()) {
    return {};
  }
  // use dfs to find the clusterable nodes.
  for (size_t i = 0; i < depends.size(); i++) {
    auto inp = base_node->input(i + 1)->cast<CNodePtr>();
    if (inp == nullptr) {
      continue;
    }
    if (depends[i] == DependOn::kValue) {
      MS_LOG(DEBUG) << "The input[" << i << "] " << inp->fullname_with_scope() << " is value-depended.";
      FindValueDependNode(inp, 1);
    } else if (IsHostOp(inp)) {
      MS_LOG(DEBUG) << "The input[" << i << "] " << inp->fullname_with_scope() << " is a host op.";
      FindValueDependNode(inp, 1);
    }
  }
  if (valid_nodes_.empty()) {
    return {};
  }
  (void)valid_nodes_.insert(base_node);
  // when the TupleGetItem's input is not in valid nodes, remove the TupleGetItem.
  RemoveWildGetitem();
  return TopoSort(base_node, SuccIncoming, [this](const AnfNodePtr &node) -> IncludeType {
    return valid_nodes_.count(node) > 0 ? FOLLOW : EXCLUDE;
  });
}

ValuePtr SymbolEngineExtender::FindOnlyDependShapeInputs(const FuncGraphPtr &fg) const {
  const auto &params = fg->parameters();
  std::vector<bool> only_depend_shape(params.size(), true);
  // depend value when infer
  for (size_t i = 0; i < params.size(); i++) {
    if (params[i]->abstract()->GetSymbolicValue() != nullptr) {
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

CNodePtr SymbolEngineExtender::CreatePacketNode(const FuncGraphPtr &main_fg, const FuncGraphPtr &sub_fg,
                                                const AnfNodePtrList &inputs) const {
  std::vector<AnfNodePtr> fn_inputs;
  fn_inputs.reserve(inputs.size() + 1);
  (void)fn_inputs.emplace_back(NewValueNode(sub_fg));
  (void)fn_inputs.insert(fn_inputs.end(), inputs.cbegin(), inputs.cend());
  auto new_cnode = main_fg->NewCNode(fn_inputs);
  new_cnode->set_abstract(sub_fg->output()->abstract());
  new_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  return new_cnode;
}

void SymbolEngineExtender::ProcessNopNode(const FuncGraphPtr &fg, AnfNodePtrList *inputs) const {
  auto real_node = fg->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_node);
  size_t idx = inputs->size();
  if (common::AnfAlgo::IsNopNode(real_node) && real_node->input(1)->isa<Parameter>()) {
    auto iter = std::find(fg->parameters().begin(), fg->parameters().end(), real_node->input(1));
    if (iter == fg->parameters().end()) {
      return;
    }
    idx = static_cast<size_t>(iter - fg->parameters().begin());
  }
  if (idx < inputs->size()) {
    fg->set_attr(kAttrNopOp, MakeValue(true));
    if (idx > 0) {
      auto new_params = fg->parameters();
      std::swap(new_params[idx], new_params[0]);
      std::swap(inputs->at(idx), inputs->at(0));
      fg->set_parameters(new_params);
    }
  }
}

void SymbolEngineExtender::ConstValueToParam(const FuncGraphPtr &fg, AnfNodePtrList *inputs) const {
  auto out_cnode = fg->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_cnode);
  for (size_t i = 1; i < out_cnode->size(); i++) {
    if (out_cnode->input(i)->isa<ValueNode>()) {
      auto vnode = out_cnode->input(i)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(vnode);
      auto p = fg->add_parameter();
      p->set_abstract(vnode->value()->ToAbstract());
      inputs->push_back(vnode);
      out_cnode->set_input(i, p);
    }
  }
}

bool SymbolEngineExtender::ExtendNode(const AnfNodePtr &node, const FuncGraphPtr &main_fg) {
  ClusterConfig config;
  config.inline_sub_func_graph = false;
  config.sort_parameter = true;

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto nodes = FindCandidates(cnode);
  visited_.clear();
  valid_nodes_.clear();
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
  ProcessNopNode(fg, &inputs);
  ConstValueToParam(fg, &inputs);
  auto symbol_engine = KernelPacketEngine::Build(fg);
  if (symbol_engine == nullptr || !symbol_engine->SupportInfer()) {
    MS_LOG(INFO) << "Symbol engine doesn't support infer shape from node: " << node->fullname_with_scope();
    return false;
  }
  auto new_cnode = CreatePacketNode(main_fg, fg, inputs);
  if (!common::AnfAlgo::IsDynamicShape(new_cnode)) {
    MS_LOG(DEBUG) << "The node " << new_cnode->DebugString() << " is not dynamic shape";
    return false;
  }
  auto fuse_op_name = GkUtils::ExtractGraphKernelName(nodes, "", "packet");
  fg->set_attr(kAttrKernelPacketNode, MakeValue(fuse_op_name));
  fg->set_attr("only_depend_shape", FindOnlyDependShapeInputs(fg));
  new_cnode->set_attrs(cnode->attrs());
  new_cnode->AddAttr(kAttrToPrim, MakeValue(AnfUtils::GetCNodeName(node) + "_packet"));
  MS_LOG(INFO) << "Replace " << node->fullname_with_scope() << " with " << new_cnode->fullname_with_scope();
  (void)main_fg->manager()->Replace(node, new_cnode);
  return true;
}

bool SymbolEngineExtender::Run(const FuncGraphPtr &func_graph) {
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
  std::reverse(cnodes.begin(), cnodes.end());
  for (auto cnode : cnodes) {
    if (!CheckBaseNode(cnode)) {
      continue;
    }
    // the node is fused.
    if (mng->node_users()[cnode].empty()) {
      continue;
    }
    if (ExtendNode(cnode, func_graph)) {
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel::packet
