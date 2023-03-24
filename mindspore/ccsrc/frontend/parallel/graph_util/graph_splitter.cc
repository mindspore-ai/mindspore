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

#include "frontend/parallel/graph_util/graph_splitter.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "mindspore/core/utils/ms_context.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/draw.h"
#include "include/common/utils/parallel_context.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_context.h"
#endif

namespace mindspore {
namespace parallel {
bool OperatorLabel::operator<(const OperatorLabel &label) const { return to_string() < label.to_string(); }

bool OperatorLabel::operator==(const OperatorLabel &label) const { return to_string() == label.to_string(); }

bool OperatorLabel::operator!=(const OperatorLabel &label) const { return !(*this == label); }

bool OperatorLabel::LooseEqual(const OperatorLabel &label, distributed::DistExecutionMode mode) const {
  if (kLabelMatchingFuncMap.count(mode) == 0) {
    MS_LOG(DEBUG) << "The mode " << mode << " does not need LooseEqual.";
    return to_string() == label.to_string();
  }
  return kLabelMatchingFuncMap.at(mode)(label, *this);
}

std::string OperatorLabel::to_string() const { return std::to_string(rank_id) + "_" + ms_role; }

ValueNodePtr CreateFakeValueNode(bool use_origin_node, const AnfNodePtr &origin_node, bool use_fake_shape) {
  tensor::TensorPtr fake_tensor = nullptr;
  if (use_origin_node) {
    MS_EXCEPTION_IF_NULL(origin_node);
    abstract::AbstractTensorPtr origin_abstract;
    if (origin_node->abstract()->isa<abstract::AbstractTuple>()) {
      // Defaultly, if the origin node's output is a tuple, get the abstract of the first element.
      auto get_one_tuple_element = origin_node->abstract()->cast<abstract::AbstractTuplePtr>()->elements()[0];
      origin_abstract = get_one_tuple_element->cast<abstract::AbstractTensorPtr>();
    } else {
      origin_abstract = origin_node->abstract()->cast<abstract::AbstractTensorPtr>();
    }
    MS_EXCEPTION_IF_NULL(origin_abstract);
    auto element = origin_abstract->element();
    MS_EXCEPTION_IF_NULL(element);
    auto build_type = element->BuildType();
    MS_EXCEPTION_IF_NULL(build_type);
    auto type_id = build_type->type_id();
    if (use_fake_shape) {
      // Assign send's output shape as {1};
      ShapeVector fake_shape = {kSizeOne};
      fake_tensor = std::make_shared<tensor::Tensor>(type_id, fake_shape);
    } else {
      auto shape = origin_abstract->shape();
      MS_EXCEPTION_IF_NULL(shape);
      fake_tensor = std::make_shared<tensor::Tensor>(type_id, shape->shape());
      fake_tensor->set_base_shape(shape->Clone());
    }
  } else {
    fake_tensor = std::make_shared<tensor::Tensor>(1.0);
  }

  MS_EXCEPTION_IF_NULL(fake_tensor);
  auto fake_value = NewValueNode(fake_tensor);
  MS_EXCEPTION_IF_NULL(fake_value);
  fake_value->set_abstract(fake_tensor->ToAbstract());
  return fake_value;
}

CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node_with_tuple_output,
                                size_t item_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_with_tuple_output);
  const auto &tuple_abstract = node_with_tuple_output->abstract();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  if (!tuple_abstract->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "Only create TupleGetItem for tuple output.";
  }

  auto item_index_value_node = NewValueNode(MakeValue(UlongToLong(item_index)));
  MS_EXCEPTION_IF_NULL(item_index_value_node);

  std::vector<AnfNodePtr> tuple_get_item_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kTupleGetItem)),
                                                   node_with_tuple_output, item_index_value_node};
  CNodePtr tuple_get_item_node = func_graph->NewCNode(tuple_get_item_inputs);
  MS_EXCEPTION_IF_NULL(tuple_get_item_node);
  tuple_get_item_node->set_abstract(tuple_abstract->cast<abstract::AbstractTuplePtr>()->elements()[item_index]);
  return tuple_get_item_node;
}

CNodePtr CreateMakeTupleNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &tuple_inputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  (void)new_make_tuple_inputs.insert(new_make_tuple_inputs.cend(), tuple_inputs.cbegin(), tuple_inputs.cend());
  auto make_tuple_node = func_graph->NewCNode(new_make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple_node);

  // MakeTuple's abstract must consist of all inputs' abstract in case unexpected graph compiling error.
  AbstractBasePtrList abstract_list;
  (void)std::for_each(tuple_inputs.cbegin(), tuple_inputs.cend(),
                      [&](const auto &input) { (void)abstract_list.emplace_back(input->abstract()); });
  make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return make_tuple_node;
}

AnfNodePtr CreateReplacedOutputNode(const FuncGraphPtr &func_graph, const AnfNodePtr &origin_output) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_output);
  MS_EXCEPTION_IF_NULL(origin_output->abstract());
  if (origin_output->abstract()->isa<abstract::AbstractTuple>()) {
    auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(origin_output, kIndex0);
    auto real_output = kernel_with_index.first;
    if (!IsPrimitiveCNode(real_output, prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION) << "Tuple output is not a MakeTuple node: " << real_output->DebugString();
    }
    AnfNodePtrList tuple_inputs;
    auto tuple_elements = origin_output->abstract()->cast<abstract::AbstractTuplePtr>()->elements();
    for (size_t i = kIndex0; i < tuple_elements.size(); i++) {
      // If tuple input is a ValueNode, use it as new tuple's input.
      const auto tuple_input = real_output->cast<CNodePtr>()->input(i + kSizeOne);
      if (tuple_input->isa<Parameter>() || tuple_input->isa<ValueNode>()) {
        MS_LOG(INFO) << "Use " << tuple_input->DebugString() << " as replaced output.";
        tuple_inputs.emplace_back(tuple_input);
        continue;
      }

      const auto &element = tuple_elements[i];
      MS_EXCEPTION_IF_NULL(element);
      auto tensor_abstract = element->cast<abstract::AbstractTensorPtr>();
      if (!tensor_abstract) {
        MS_LOG(EXCEPTION) << "Only support to replace tuple with all tensor elements.";
      }
      auto fake_tensor = std::make_shared<tensor::Tensor>(tensor_abstract->element()->BuildType()->type_id(),
                                                          tensor_abstract->shape()->shape());
      MS_EXCEPTION_IF_NULL(fake_tensor);
      auto fake_value_node = NewValueNode(fake_tensor);
      MS_EXCEPTION_IF_NULL(fake_value_node);
      fake_value_node->set_abstract(fake_tensor->ToAbstract());
      (void)tuple_inputs.emplace_back(fake_value_node);
    }
    return CreateMakeTupleNode(func_graph, tuple_inputs);
  } else {
    return CreateFakeValueNode(true, origin_output);
  }
}

void SetSendNodeAttr(const AnfNodePtr &send_node, const InterProcessOpEdge &inter_process_edge) {
  const auto &send_src_node = inter_process_edge.src_node;
  const auto &send_dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(send_src_node);
  MS_EXCEPTION_IF_NULL(send_dst_node);
  MS_EXCEPTION_IF_NULL(send_node);

  std::string src_node_name = send_src_node->fullname_with_scope();
  std::string dst_node_name = send_dst_node->fullname_with_scope();

  // These attributes are the inter-process edge information.
  std::vector<uint32_t> dst_ranks = {inter_process_edge.dst_label.rank_id};
  common::AnfAlgo::SetNodeAttr(kAttrSendDstRanks, MakeValue(dst_ranks), send_node);
  std::vector<std::string> dst_roles = {inter_process_edge.dst_label.ms_role};
  common::AnfAlgo::SetNodeAttr(kAttrSendDstRoles, MakeValue(dst_roles), send_node);

  common::AnfAlgo::SetNodeAttr(kAttrSendSrcNodeName, MakeValue(src_node_name), send_node);
  common::AnfAlgo::SetNodeAttr(kAttrSendDstNodeName, MakeValue(dst_node_name), send_node);
  std::vector<std::string> inter_process_edges = {inter_process_edge.to_string()};
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(inter_process_edges), send_node);

  // Set send node to CPU for now.
  common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice), send_node);
}

void SetRecvNodeAttr(const AnfNodePtr &recv_node, const InterProcessOpEdge &inter_process_edge) {
  const auto &recv_src_node = inter_process_edge.src_node;
  const auto &recv_dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(recv_src_node);
  MS_EXCEPTION_IF_NULL(recv_dst_node);
  MS_EXCEPTION_IF_NULL(recv_node);

  std::string src_node_name = recv_src_node->fullname_with_scope();
  std::string dst_node_name = recv_dst_node->fullname_with_scope();

  // These attributes are the inter-process edge information.
  std::vector<uint32_t> src_ranks = {inter_process_edge.src_label.rank_id};
  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRanks, MakeValue(src_ranks), recv_node);
  std::vector<std::string> src_roles = {inter_process_edge.src_label.ms_role};
  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcRoles, MakeValue(src_roles), recv_node);

  common::AnfAlgo::SetNodeAttr(kAttrRecvSrcNodeName, MakeValue(src_node_name), recv_node);
  common::AnfAlgo::SetNodeAttr(kAttrRecvDstNodeName, MakeValue(dst_node_name), recv_node);
  std::vector<std::string> inter_process_edges = {inter_process_edge.to_string()};
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(inter_process_edges), recv_node);

  // Set recv node to CPU for now.
  common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice), recv_node);
}

CNodePtr CreateSendNode(const FuncGraphPtr &func_graph, const InterProcessOpEdge &inter_process_edge) {
  const auto &src_node = inter_process_edge.src_node;
  const auto &dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);

  std::vector<AnfNodePtr> send_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcSendOpName))};
  ValueNodePtr mock_value = nullptr;
  if (IsPrimitiveCNode(src_node, prim::kPrimUpdateState)) {
    mock_value = CreateFakeValueNode(false);
    send_inputs.push_back(mock_value);
    send_inputs.push_back(src_node);
  } else {
    if (src_node->abstract()->isa<abstract::AbstractTuple>()) {
      // If src_node's output is a tuple, get the first element of the tuple as Send's input.
      auto tuple_get_item_node = CreateTupleGetItemNode(func_graph, src_node, kIndex0);
      send_inputs.push_back(tuple_get_item_node);
      mock_value = CreateFakeValueNode(true, tuple_get_item_node);
    } else {
      send_inputs.push_back(src_node);
      mock_value = CreateFakeValueNode(true, src_node);
    }
  }
  CNodePtr send_node = func_graph->NewCNode(send_inputs);
  MS_EXCEPTION_IF_NULL(send_node);
  send_node->set_abstract(mock_value->abstract());

  SetSendNodeAttr(send_node, inter_process_edge);
  return send_node;
}

CNodePtr CreateRecvNode(const FuncGraphPtr &func_graph, const InterProcessOpEdge &inter_process_edge) {
  const auto &src_node = inter_process_edge.src_node;
  const auto &dst_node = inter_process_edge.dst_node;
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);

  std::vector<AnfNodePtr> recv_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcRecvOpName))};
  CNodePtr recv_node = nullptr;
  AbstractBasePtr recv_node_abs = nullptr;
  if (IsPrimitiveCNode(src_node, prim::kPrimUpdateState)) {
    ValuePtr monad_value = nullptr;
    if (HasAbstractUMonad(src_node)) {
      monad_value = kUMonad;
    } else if (HasAbstractIOMonad(src_node)) {
      monad_value = kIOMonad;
    } else {
      MS_LOG(EXCEPTION) << "The src_node is PrimUpdateState must have monad abstract.";
    }
    auto monad_input = NewValueNode(monad_value);
    MS_EXCEPTION_IF_NULL(monad_input);
    monad_input->set_abstract(monad_value->ToAbstract());
    recv_inputs.push_back(monad_input);
    recv_node_abs = src_node->abstract();
  } else {
    if (src_node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrUpdateParameter, src_node->cast<CNodePtr>()) &&
        common::AnfAlgo::HasNodeAttr(kAttrParameterInputIndex, src_node->cast<CNodePtr>())) {
      int64_t parameter_index = common::AnfAlgo::GetNodeAttr<int64_t>(src_node, kAttrParameterInputIndex);
      auto kernel_with_index = common::AnfAlgo::VisitKernel(
        common::AnfAlgo::GetInputNode(src_node->cast<CNodePtr>(), parameter_index), kIndex0);
      auto param_node = kernel_with_index.first;
      recv_inputs.push_back(param_node);

      // To update the parameter on the device side in heterogeneous case, side-effect node should be added to recv's
      // input.
      ValuePtr monad_value = kUMonad;
      auto monad_input = NewValueNode(monad_value);
      MS_EXCEPTION_IF_NULL(monad_input);
      monad_input->set_abstract(monad_value->ToAbstract());
      recv_inputs.push_back(monad_input);

      recv_node_abs = param_node->abstract();
    } else if (src_node->isa<CNode>() && common::AnfAlgo::GetCNodeName(src_node) == distributed::kDataSyncSrcOpName) {
      auto kernel_with_index =
        common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(src_node->cast<CNodePtr>(), kIndex0), kIndex0);
      auto param_node = kernel_with_index.first;
      recv_inputs.push_back(param_node);

      ValuePtr monad_value = kUMonad;
      auto monad_input = NewValueNode(monad_value);
      MS_EXCEPTION_IF_NULL(monad_input);
      monad_input->set_abstract(monad_value->ToAbstract());
      recv_inputs.push_back(monad_input);

      recv_node_abs = param_node->abstract();
    } else {
      // Use the same shape as origin node's.
      auto mock_value = CreateFakeValueNode(true, src_node, false);
      MS_EXCEPTION_IF_NULL(mock_value);
      recv_inputs.push_back(mock_value);
      recv_node_abs = src_node->abstract();
    }
  }
  recv_node = func_graph->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(recv_node);
  recv_node->set_abstract(recv_node_abs);

  SetRecvNodeAttr(recv_node, inter_process_edge);
  return recv_node;
}

std::map<size_t, size_t> GetRealIndexToSeg(const std::vector<size_t> &split_segment, size_t real_size) {
  std::map<size_t, size_t> result;
  // If split_segment is empty, return an empty map.
  if (split_segment.empty()) {
    return result;
  }

  // Check whether the vector of indices is valid.
  if (!std::is_sorted(split_segment.begin(), split_segment.end())) {
    MS_LOG(EXCEPTION) << "Indices of segments is not in a ascending order: " << split_segment;
  }

  size_t real_index = 0;
  for (size_t seg_index = 0; seg_index < split_segment.size(); seg_index++) {
    size_t upper_bound = split_segment[seg_index];
    for (; real_index < real_size; real_index++) {
      if (real_index <= upper_bound) {
        result[real_index] = seg_index;
      } else {
        break;
      }
    }
  }

  // Map the rest of real index to a segment.
  if (real_size > (*split_segment.rbegin()) + 1) {
    for (; real_index < real_size; real_index++) {
      result[real_index] = split_segment.size();
    }
  }
  return result;
}

bool IsOneOfRealGraphInput(const FuncGraphPtr &func_graph, const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  auto all_inputs = func_graph->get_inputs();
  return std::count(all_inputs.begin(), all_inputs.end(), input) != 0;
}

distributed::DistExecutionMode GenerateStrategy() {
  distributed::DistExecutionMode strategy;
  bool enable_ps = false;
  bool enable_embedding_cache = false;
#if defined(__linux__) && defined(WITH_BACKEND)
  enable_ps = ps::PSContext::instance()->is_ps_mode();
  enable_embedding_cache = ps::PSContext::instance()->cache_enable();
#endif
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  MS_LOG(INFO) << "Current parallel mode is " << parallel_mode;
  bool using_parallel = (parallel_mode != parallel::kStandalone) ? true : false;
  // The conditions' priority is: EmbeddingCache > Parameter Server > General.
  if (enable_embedding_cache) {
    strategy = distributed::DistExecutionMode::kEmbeddingCacheMode;
  } else if (enable_ps) {
    strategy = distributed::DistExecutionMode::kPSMode;
  } else if (using_parallel) {
    strategy = distributed::DistExecutionMode::kParallelMode;
  } else {
    strategy = distributed::DistExecutionMode::kGeneralMode;
  }
  MS_LOG(INFO) << "Generated distributed strategy is " << strategy;
  return strategy;
}

void TransformPrimAttrToAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(kIndex0));
  MS_EXCEPTION_IF_NULL(prim);
  if (cnode->HasPrimalAttr(distributed::kOpLabelRankId)) {
    MS_LOG(DEBUG) << cnode->fullname_with_scope() << " has primal attr 'rank_id'.";
    prim->set_attr(distributed::kOpLabelRankId, cnode->GetPrimalAttr(distributed::kOpLabelRankId));
  }
  if (cnode->HasPrimalAttr(distributed::kOpLabelRole)) {
    MS_LOG(DEBUG) << cnode->fullname_with_scope() << " has primal attr 'ms_role'.";
    prim->set_attr(distributed::kOpLabelRole, cnode->GetPrimalAttr(distributed::kOpLabelRole));
  }
}

bool NodeHasLabel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }

  bool has_label = false;
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim_node = cnode->input(0);
  MS_EXCEPTION_IF_NULL(prim_node);

  // As long as the node has 'ms_role' and 'rank_id' attributes, we consider this node has label regardless the value of
  // these two attributes.
  if (IsValueNode<Primitive>(prim_node)) {
    auto prim = GetValueNode<PrimitivePtr>(prim_node);
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->HasAttr(distributed::kOpLabelRankId) && prim->HasAttr(distributed::kOpLabelRole)) {
      has_label = true;
    }
  } else {
    // Get label for call node, 'call' node hasn't primitive to save attrs, so get attrs of 'call' from cnode.
    if (cnode->HasAttr(distributed::kOpLabelRankId) && cnode->HasAttr(distributed::kOpLabelRole)) {
      has_label = true;
    }
  }
  return has_label;
}

bool GraphHasLabel(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph->get_return());
  // If one node has label, this graph has label. Thus it needs to be split.
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (NodeHasLabel(node)) {
      return true;
    }
  }
  return false;
}

CNodePtrList GetSideEffectNodeList(const AnfNodePtrList &nodes) {
  CNodePtrList side_effect_nodes;
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM)) {
      side_effect_nodes.emplace_back(cnode);
      MS_LOG(DEBUG) << "CNode with side effect mem: " << cnode->fullname_with_scope();
    }
  }
  return side_effect_nodes;
}

AnfNodePtrList GetRefInputs(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtrList ref_inputs;
  for (size_t i = kIndex1; i < cnode->size(); ++i) {
    auto &input = cnode->inputs().at(i);
    if (common::AnfAlgo::HasAbstractRef(input)) {
      ref_inputs.push_back(input);
      MS_LOG(DEBUG) << "The ref input " << input->fullname_with_scope() << " of node " << cnode->fullname_with_scope();
    }
  }
  return ref_inputs;
}

CNodePtr FindNextUpdateStateNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto cnode_users = func_graph->manager()->node_users()[cnode];
  for (const auto &user : cnode_users) {
    auto user_node = user.first;
    MS_EXCEPTION_IF_NULL(user_node);
    if (common::AnfAlgo::GetCNodeName(user_node) == kUpdateStateOpName) {
      return user_node->cast<CNodePtr>();
    }
  }
  return nullptr;
}

ValueNodePtr CreateUMonadNode() {
  ValuePtr monad_value = kUMonad;
  auto monad_input = NewValueNode(monad_value);
  MS_EXCEPTION_IF_NULL(monad_input);
  monad_input->set_abstract(monad_value->ToAbstract());
  return monad_input;
}

CNodePtr CreateUpdateStateNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &update_state_inputs) {
  if (update_state_inputs.empty()) {
    MS_LOG(EXCEPTION) << "The inputs of UpdateState should not be empty.";
  }
  // The first input of UpdateState is an 'U'.
  ValueNodePtr umonad_input = CreateUMonadNode();
  MS_EXCEPTION_IF_NULL(umonad_input);
  AnfNodePtrList inputs = {NewValueNode(prim::kPrimUpdateState), umonad_input};
  inputs.insert(inputs.end(), update_state_inputs.begin(), update_state_inputs.end());

  auto update_state_node = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(update_state_node);
  update_state_node->set_abstract(umonad_input->abstract());
  return update_state_node;
}

std::map<AnfNodePtr, AnfNodePtrSet> FilterDependencyToTargetNode(const FuncGraphPtr &func_graph,
                                                                 const AnfNodePtrSet &target_nodes) {
  std::map<AnfNodePtr, AnfNodePtrSet> depend_matrix;
  MS_EXCEPTION_IF_NULL(func_graph);
  auto return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  AnfNodePtrList nodes = FuncGraph::TopoSort(return_node);
  // Trasverse all nodes in topo-sort so that time complexity is O(n).
  for (const auto node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    CNodePtr cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    // Traverse all inputs and only filter out inputs which is in target nodes set.
    for (const auto &input : inputs) {
      MS_EXCEPTION_IF_NULL(input);
      // If the input is stored already, this means it depends on some of target nodes, so we expand its inputs and
      // insert them.
      if (depend_matrix.count(input) != 0) {
        depend_matrix[node].insert(depend_matrix[input].begin(), depend_matrix[input].end());
      }
      // If input itself is in target nodes set, insert it as well.
      if (target_nodes.count(input) != 0) {
        depend_matrix[node].insert(input);
      }
    }
  }
  return depend_matrix;
}

AnfNodePtrSet UpdateDependedSet(const AnfNodePtr &new_node, const AnfNodePtrSet &old_depended_set,
                                const std::map<AnfNodePtr, AnfNodePtrSet> &node_dependency) {
  AnfNodePtrSet updated = old_depended_set;
  bool is_independent = true;
  for (const auto &stored_node : old_depended_set) {
    // If 'new_node' is already depended on by 'stored_node', no need to add 'new_node'.
    if (node_dependency.count(stored_node) != 0 && node_dependency.at(stored_node).count(new_node) != 0) {
      MS_LOG(DEBUG) << "Old node " << stored_node->fullname_with_scope() << " depends on "
                    << new_node->fullname_with_scope() << ". Do not update.";
      is_independent = false;
      break;
    }
    // If 'new_node' depends on 'stored_node', replace 'stored_node' with 'new_node' to keep minimal dependency.
    if (node_dependency.count(new_node) != 0 && node_dependency.at(new_node).count(stored_node) != 0) {
      MS_LOG(DEBUG) << "Replace old node " << stored_node->fullname_with_scope() << " with new node "
                    << new_node->fullname_with_scope();
      updated.erase(stored_node);
      updated.insert(new_node);
    }
  }
  if (is_independent) {
    MS_LOG(DEBUG) << "Add new node to depended set " << new_node->fullname_with_scope();
    updated.insert(new_node);
  }
  return updated;
}

void HandleHungNodes(const FuncGraphPtr &func_graph, const NodeLabels &node_labels, OperatorLabel process_label,
                     const AnfNodePtrList &hung_nodes_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto make_tuple_node = CreateMakeTupleNode(func_graph, hung_nodes_list);
  MS_EXCEPTION_IF_NULL(make_tuple_node);

  const auto &origin_output = func_graph->output();
  MS_EXCEPTION_IF_NULL(origin_output);
  if (node_labels.count(origin_output) == 0) {
    MS_LOG(EXCEPTION) << "The origin output node " << origin_output->fullname_with_scope()
                      << " should have corresponding operator label.";
  }
  AnfNodePtr replaced_output = nullptr;
  if (node_labels.at(origin_output) != process_label) {
    replaced_output = CreateReplacedOutputNode(func_graph, origin_output);
  } else {
    replaced_output = origin_output;
  }
  MS_EXCEPTION_IF_NULL(replaced_output);

  // Add dependency and replace.
  std::vector<AnfNodePtr> depend_inputs = {NewValueNode(prim::kPrimDepend), replaced_output, make_tuple_node};
  auto final_output_node = func_graph->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(final_output_node);
  final_output_node->set_abstract(replaced_output->abstract());
  (void)func_graph->manager()->SetEdge(func_graph->get_return(), 1, final_output_node);
}

void ParameterServerMode::PreBuildDistributedGraph() {
  MS_LOG(INFO) << "Start pre-building distribtued graph in Parameter Server mode.";
  MS_EXCEPTION_IF_NULL(node_labels_);
  ProcessForSplitOptimizer();
  MS_LOG(INFO) << "End pre-building distribtued graph in Parameter Server mode.";
}

FusedInterProcessOpPairMap ParameterServerMode::DoRpcNodeFusion(InterProcessOpEdgesInfo *comm_edges_ptr) {
  MS_EXCEPTION_IF_NULL(comm_edges_ptr);

  // The edges of server optimizers should be fused with same peers. For example, edges from Worker_0 to Server_0 will
  // be fused by segments.
  InterProcessOpEdgesInfo comm_edges_of_server_optimizer = FilterCommEdgesOfServerOptimizer(*comm_edges_ptr);
  FusedInterProcessOpPairMap optimizer_fused_edges = FuseRpcNodesForSplitOptimizer(comm_edges_of_server_optimizer);

  // The rest of the edges are not fused like edges for EmbeddingLookup, but the FusedInterProcessOpPairMap object
  // should be created.
  FusedInterProcessOpPairMap rest_edges = FilterNotServerOptimizerEdges(*comm_edges_ptr);
  optimizer_fused_edges.insert(rest_edges.cbegin(), rest_edges.cend());
  return optimizer_fused_edges;
}

void ParameterServerMode::PostBuildDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) {
  MS_LOG(INFO) << "Start post-building distribtued graph in Parameter Server mode.";
  MS_EXCEPTION_IF_NULL(node_labels_);
  // Judge the node role number validation.
  uint32_t worker_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfWorker);
  if (worker_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, worker number should be greater than 0.";
  }
  uint32_t server_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfServer);
  if (server_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, server number should be greater than 0.";
  }
  // Only multiple worker scenario needs this optimizer.
  if (worker_num < kMinGradAccumWorkerNum) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<CNodePtr> ps_optimizer_node_list = FilterServerAwareOptimizerList();

  // Duplicate out degrees for ps optimizers because defaultly there's only one edge to the rank 0 worker.
  for (const auto &ps_optimizer : ps_optimizer_node_list) {
    for (const auto &edge_info : comm_edges) {
      if (edge_info.first.src_node == ps_optimizer) {
        // The optimizer's output should always connect to Send node which is the input of a MakeTuple node.
        // We need to replace the MakeTuple node with a new one.
        const auto &origin_send_node = std::get<0>(edge_info.second);
        std::vector<AnfNodePtr> new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), origin_send_node};
        AnfNodePtr dst_node = edge_info.first.dst_node;
        for (uint32_t i = 1; i < worker_num; i++) {
          OperatorLabel worker_label = {i, distributed::kEnvRoleOfWorker};
          InterProcessOpEdge edge = {ps_optimizer, node_labels_->at(ps_optimizer), dst_node, worker_label};
          auto duplicated_send_node = CreateSendNode(func_graph_, edge);
          (void)node_labels_->insert(std::make_pair(duplicated_send_node, edge.src_label));
          (void)new_make_tuple_inputs.emplace_back(duplicated_send_node);
        }
        auto new_make_tuple_node = func_graph_->NewCNode(new_make_tuple_inputs);
        new_make_tuple_node->set_abstract(new_make_tuple_inputs.back()->abstract());
        (void)func_graph_->manager()->Replace(origin_send_node, new_make_tuple_node);
      }
    }
  }
  MS_LOG(INFO) << "End post-building distribtued graph in Parameter Server mode.";
}

void ParameterServerMode::PostBuildDistributedGraph(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs) {
  MS_LOG(INFO) << "Start post-building distribtued graph in Parameter Server mode.";
  MS_EXCEPTION_IF_NULL(node_labels_);
  // Judge the node role number validation.
  uint32_t worker_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfWorker);
  if (worker_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, worker number should be greater than 0.";
  }
  uint32_t server_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfServer);
  if (server_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, server number should be greater than 0.";
  }
  // Only multiple worker scenario needs this optimizer.
  if (worker_num < kMinGradAccumWorkerNum) {
    return;
  }

  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<CNodePtr> ps_optimizer_node_list = FilterServerAwareOptimizerList();
  if (ps_optimizer_node_list.empty()) {
    MS_LOG(INFO) << "This process has no ps optimizer on it. No need to do post building.";
    return;
  }

  // Duplicate out degrees for ps optimizers because defaultly there's only one edge to the rank 0 worker.
  for (const auto &op_pair_info : fused_inter_process_op_pairs) {
    const auto &op_pairs = op_pair_info.second;
    CNodePtr fused_send_node = std::get<0>(op_pairs[0]);
    // Node's inputs except primtive value node.
    std::vector<AnfNodePtr> fused_send_node_inputs = fused_send_node->inputs();
    (void)fused_send_node_inputs.erase(fused_send_node_inputs.cbegin());

    // Only handle the edge whose src_node is optimizer.
    if (std::find_if(ps_optimizer_node_list.cbegin(), ps_optimizer_node_list.cend(), [&](const auto &ps_optimizer) {
          return ps_optimizer.get() == fused_send_node_inputs[0].get();
        }) == ps_optimizer_node_list.cend()) {
      continue;
    }

    std::vector<AnfNodePtr> new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), fused_send_node};
    for (uint32_t i = 1; i < worker_num; i++) {
      std::vector<CNodePtr> new_send_nodes;
      OperatorLabel worker_label = {i, distributed::kEnvRoleOfWorker};
      for (size_t j = 0; j < op_pairs.size(); j++) {
        const auto &src_node = fused_send_node_inputs[j];
        const auto &dst_node = std::get<3>(op_pairs[j]);
        InterProcessOpEdge edge = {src_node, node_labels_->at(src_node), dst_node, worker_label};
        auto duplicated_send_node = CreateSendNode(func_graph_, edge);
        MS_EXCEPTION_IF_NULL(duplicated_send_node);
        (void)node_labels_->insert(std::make_pair(duplicated_send_node, edge.src_label));
        (void)new_send_nodes.emplace_back(duplicated_send_node);
      }
      CNodePtr new_fused_send_node = FuseRpcSendNodes(new_send_nodes);
      MS_EXCEPTION_IF_NULL(new_fused_send_node);
      (void)new_make_tuple_inputs.emplace_back(new_fused_send_node);
    }
    auto new_make_tuple_node = func_graph_->NewCNode(new_make_tuple_inputs);
    new_make_tuple_node->set_abstract(fused_send_node->abstract());
    (void)func_graph_->manager()->Replace(fused_send_node, new_make_tuple_node);
  }
  MS_LOG(INFO) << "End post-building distribtued graph in Parameter Server mode.";
}

void ParameterServerMode::ProcessForSplitOptimizer() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<CNodePtr> ps_optimizer_node_list = FilterServerAwareOptimizerList();

  // Judge the node role number validation.
  uint32_t worker_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfWorker);
  if (worker_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, worker number should be greater than 0.";
  }
  uint32_t server_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfServer);
  if (server_num == 0) {
    MS_LOG(EXCEPTION) << "In PS mode, server number should be greater than 0.";
  }
  // Only multiple worker scenario needs this optimizer.
  if (worker_num < kMinGradAccumWorkerNum) {
    return;
  }

  for (const auto &ps_optimizer : ps_optimizer_node_list) {
    MS_EXCEPTION_IF_NULL(ps_optimizer);
    // Load attributes for this optimizer.
    auto gradient_index = common::AnfAlgo::HasNodeAttr(kAttrGradientInputIndex, ps_optimizer)
                            ? LongToSize(common::AnfAlgo::GetNodeAttr<int64_t>(ps_optimizer, kAttrGradientInputIndex))
                            : UINT64_MAX;
    size_t indices_index = common::AnfAlgo::HasNodeAttr(kAttrIndicesInputIndex, ps_optimizer)
                             ? LongToSize(common::AnfAlgo::GetNodeAttr<int64_t>(ps_optimizer, kAttrIndicesInputIndex))
                             : UINT64_MAX;
    std::string gradient_type = (common::AnfAlgo::HasNodeAttr(kAttrGradientType, ps_optimizer))
                                  ? common::AnfAlgo::GetNodeAttr<std::string>(ps_optimizer, kAttrGradientType)
                                  : kDenseGradient;
    if (kGradTypeToAccumOpName.count(gradient_type) == 0) {
      MS_LOG(EXCEPTION) << "The gradient type " << gradient_type << " is invalid.";
    }

    const std::string &opt_device_target = GetCNodeTarget(ps_optimizer);
    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(ps_optimizer); i++) {
      auto input = common::AnfAlgo::GetInputNode(ps_optimizer, i);
      // If the input is not a cnode, no inter-process edge is added so no node with multiple inputs should be created.
      // Unless it's a real input.
      if (!input->isa<CNode>()) {
        if (IsOneOfRealGraphInput(func_graph_, input)) {
          MS_LOG(INFO) << "The input " << i << " of optimizer " << ps_optimizer->fullname_with_scope() << ": "
                       << input->fullname_with_scope() << " is a real input from data.";
        } else {
          continue;
        }
      }

      if (i == gradient_index) {
        // Create the node to replace origin gradient which could be a RealDiv node.
        std::pair<CNodePtr, CNodePtr> grad_accum_nodes = CreateNodesForGradAccumulation(
          input, (role_ == distributed::kEnvRoleOfWorker) ? rank_id_ : 0, gradient_type, worker_num);

        const auto &accum_node = grad_accum_nodes.first;
        const auto &real_div_node = grad_accum_nodes.second;
        func_graph_->manager()->SetEdge(ps_optimizer, i + 1, real_div_node);
        (void)node_labels_->insert(std::make_pair(accum_node, node_labels_->at(ps_optimizer)));
        (void)node_labels_->insert(std::make_pair(real_div_node, node_labels_->at(ps_optimizer)));
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), accum_node);
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), real_div_node);
        common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeLabel, MakeValue(kPSOptimizerEdgeLabel), accum_node);
        common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeLabel, MakeValue(kPSOptimizerEdgeLabel), real_div_node);
      } else if (i == indices_index) {
        // Create the node to replace origin indices.
        AnfNodePtr new_indices_input = CreateNodeWithInterProcessEdgeOnPServer(
          kConcatOpName, input, (role_ == distributed::kEnvRoleOfWorker) ? rank_id_ : 0, worker_num);

        func_graph_->manager()->SetEdge(ps_optimizer, i + 1, new_indices_input);
        (void)node_labels_->insert(std::make_pair(new_indices_input, node_labels_->at(ps_optimizer)));
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), new_indices_input);
        common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeLabel, MakeValue(kPSOptimizerEdgeLabel), new_indices_input);
      } else {
        std::pair<CNodePtr, CNodePtr> make_tuple_get_item_nodes = CreateNodesForMakeTuple(input, worker_num);

        auto &make_tuple_node = make_tuple_get_item_nodes.first;
        auto &tuple_get_item_node = make_tuple_get_item_nodes.second;
        func_graph_->manager()->SetEdge(ps_optimizer, i + 1, tuple_get_item_node);
        (void)node_labels_->insert(std::make_pair(make_tuple_node, node_labels_->at(ps_optimizer)));
        (void)node_labels_->insert(std::make_pair(tuple_get_item_node, node_labels_->at(ps_optimizer)));
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), make_tuple_node);
        common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(opt_device_target), tuple_get_item_node);
        common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeLabel, MakeValue(kPSOptimizerEdgeLabel), make_tuple_node);
        common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeLabel, MakeValue(kPSOptimizerEdgeLabel), tuple_get_item_node);
      }
    }
  }
}

std::vector<CNodePtr> ParameterServerMode::FilterServerAwareOptimizerList() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  auto return_node = func_graph_->get_return();
  MS_EXCEPTION_IF_NULL(return_node);

  std::vector<CNodePtr> ps_optim_list;
  std::vector<AnfNodePtr> nodes = FuncGraph::TopoSort(return_node);
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    if (common::AnfAlgo::HasNodeAttr(kAttrUpdateParameter, cnode)) {
      (void)ps_optim_list.emplace_back(cnode);
      common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeLabel, MakeValue(kPSOptimizerEdgeLabel), cnode);
    }
  }
  return ps_optim_list;
}

std::pair<CNodePtr, CNodePtr> ParameterServerMode::CreateNodesForGradAccumulation(const AnfNodePtr &gradient_input,
                                                                                  size_t gradient_input_index,
                                                                                  const std::string &gradient_type,
                                                                                  size_t total_gradient_number) {
  MS_EXCEPTION_IF_NULL(gradient_input);

  if (kGradTypeToAccumOpName.count(gradient_type) == 0) {
    MS_LOG(EXCEPTION) << "The gradient type " << gradient_type << " is invalid.";
  }
  const std::string &accum_node_name = kGradTypeToAccumOpName.at(gradient_type);
  CNodePtr grad_accum_node = CreateNodeWithInterProcessEdgeOnPServer(accum_node_name, gradient_input,
                                                                     gradient_input_index, total_gradient_number);
  MS_EXCEPTION_IF_NULL(grad_accum_node);

  CNodePtr grad_mean_node = CreateGradMeanNode(grad_accum_node, total_gradient_number);
  MS_EXCEPTION_IF_NULL(grad_mean_node);
  return std::make_pair(grad_accum_node, grad_mean_node);
}

CNodePtr ParameterServerMode::CreateGradMeanNode(const AnfNodePtr &gradient, size_t divisor) {
  MS_EXCEPTION_IF_NULL(gradient);

  // Step 1: Create the value node of divisor. The divisor's value is worker number.
  auto addn_abstract = gradient->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(addn_abstract);
  // Use reciprocal of the divisor so Mul node should be created.
  auto divisor_tensor =
    std::make_shared<tensor::Tensor>(1 / static_cast<double>(divisor), addn_abstract->element()->BuildType());
  MS_EXCEPTION_IF_NULL(divisor_tensor);
  auto divisor_value_node = NewValueNode(divisor_tensor);
  MS_EXCEPTION_IF_NULL(divisor_value_node);
  divisor_value_node->set_abstract(divisor_tensor->ToAbstract());

  // Step 2: Create Mul node.
  std::vector<AnfNodePtr> real_div_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)), gradient,
                                             divisor_value_node};
  CNodePtr grad_mean_node = func_graph_->NewCNode(real_div_inputs);
  MS_EXCEPTION_IF_NULL(grad_mean_node);
  grad_mean_node->set_abstract(gradient->abstract());
  return grad_mean_node;
}

std::pair<CNodePtr, CNodePtr> ParameterServerMode::CreateNodesForMakeTuple(const AnfNodePtr &input,
                                                                           size_t total_inputs_number) {
  MS_EXCEPTION_IF_NULL(input);
  CNodePtr make_tuple_node = CreateNodeWithInterProcessEdgeOnPServer(
    prim::kMakeTuple, input, (role_ == distributed::kEnvRoleOfWorker) ? rank_id_ : 0, total_inputs_number);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  // For MakeTuple node on Parameter Server, we get the first input as its abstract because the other inputs are
  // supposed to be the same as the first one.
  CNodePtr tuple_get_item_node = CreateTupleGetItemNode(func_graph_, make_tuple_node, kIndex0);
  return std::make_pair(make_tuple_node, tuple_get_item_node);
}

CNodePtr ParameterServerMode::CreateNodeWithInterProcessEdgeOnPServer(const std::string &many_to_one_node_name,
                                                                      const AnfNodePtr &real_input,
                                                                      size_t index_of_real_input,
                                                                      uint32_t total_inputs_number) {
  if (index_of_real_input >= total_inputs_number) {
    MS_LOG(EXCEPTION) << "The index of real input for " << many_to_one_node_name << " " << index_of_real_input
                      << " is greater or equal to worker number " << total_inputs_number;
  }

  // Step 1: Create multiple inputs of new node including extra nodes.
  std::vector<AnfNodePtr> new_node_inputs;
  new_node_inputs.resize(total_inputs_number);
  std::vector<AnfNodePtr> mock_node_inputs = {NewValueNode(std::make_shared<Primitive>(
    IsPrimitiveCNode(real_input, prim::kPrimUpdateState) ? kUpdateStateOpName : kVirtualNode))};
  for (size_t i = 0; i < new_node_inputs.size(); i++) {
    new_node_inputs[i] = func_graph_->NewCNode(mock_node_inputs);
    MS_EXCEPTION_IF_NULL(new_node_inputs[i]);
    new_node_inputs[i]->set_abstract(real_input->abstract());
    new_node_inputs[i]->cast<CNodePtr>()->set_fullname_with_scope(real_input->fullname_with_scope());

    // Set operator label for new node's inputs.
    OperatorLabel input_label = {SizeToUint(i), distributed::kEnvRoleOfWorker};
    (void)node_labels_->insert(std::make_pair(new_node_inputs[i], input_label));
  }
  new_node_inputs[index_of_real_input] = real_input;

  // Step 2: Create the new node.
  auto new_node_prim = NewValueNode(std::make_shared<Primitive>(many_to_one_node_name));
  (void)new_node_inputs.insert(new_node_inputs.cbegin(), new_node_prim);

  auto new_node = func_graph_->NewCNode(new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);

  // Step 3: Set the new node's abstract and attrs.
  if (many_to_one_node_name == kAddNOpName) {
    common::AnfAlgo::SetNodeAttr("N", MakeValue(static_cast<int64_t>(total_inputs_number)), new_node);
    common::AnfAlgo::SetNodeAttr("n", MakeValue(static_cast<int64_t>(total_inputs_number)), new_node);
    new_node->set_abstract(real_input->abstract());
  } else if (many_to_one_node_name == kConcatOpName) {
    auto origin_abs = real_input->abstract()->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(origin_abs);

    auto new_abs = origin_abs->Clone()->cast<abstract::AbstractTensorPtr>();
    ShapeVector new_shape = new_abs->shape()->shape();
    new_shape[0] = new_shape[0] * static_cast<int64_t>(total_inputs_number);
    new_abs->shape()->set_shape(new_shape);
    new_node->set_abstract(new_abs);

    // Concat node must have attribute "axis" or kernel building will fail.
    size_t axis_index = 0;
    common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(UlongToLong(axis_index)), new_node);
  } else if (many_to_one_node_name == prim::kMakeTuple) {
    AbstractBasePtrList abstract_list;
    auto first_input = new_node_inputs.begin();
    std::advance(first_input, 1);
    (void)std::for_each(first_input, new_node_inputs.end(),
                        [&](const auto &input) { (void)abstract_list.emplace_back(input->abstract()); });
    new_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  } else {
    new_node->set_abstract(real_input->abstract());
  }
  return new_node;
}

FusedInterProcessOpPairMap ParameterServerMode::FuseRpcNodesForSplitOptimizer(
  const InterProcessOpEdgesInfo &comm_edges_of_server_optimizer) {
  InterProcessOpPairMap comm_edges_with_same_peer;
  for (const auto &comm_edge_info : comm_edges_of_server_optimizer) {
    const InterProcessOpEdge &edge = comm_edge_info.first;
    const InterProcessOpPair &node_pair = comm_edge_info.second;
    (void)comm_edges_with_same_peer[{edge.src_label, edge.dst_label, 0}].emplace_back(node_pair);
  }

  InterProcessOpPairMap comm_edges_segments;
  for (auto comm_edge_info = comm_edges_with_same_peer.cbegin(); comm_edge_info != comm_edges_with_same_peer.cend();
       ++comm_edge_info) {
    InterProcessEdgeWithIndex edge_with_index = comm_edge_info->first;
    const std::vector<InterProcessOpPair> &op_pair_list = comm_edge_info->second;
    std::map<size_t, size_t> real_index_to_segment =
      GetRealIndexToSeg(ps_optimizer_fusion_segments_, op_pair_list.size());
    if (real_index_to_segment.empty()) {
      comm_edges_segments[edge_with_index] = op_pair_list;
      continue;
    } else {
      if (real_index_to_segment.size() != op_pair_list.size()) {
        MS_LOG(EXCEPTION) << "Real index to segment index map is invalid: size not matched.";
      }
      for (size_t i = 0; i < op_pair_list.size(); i++) {
        edge_with_index.index = real_index_to_segment[i];
        (void)comm_edges_segments[edge_with_index].emplace_back(op_pair_list[i]);
      }
    }
  }

  FusedInterProcessOpPairMap results;
  for (auto rpc_nodes_fuse_info = comm_edges_segments.begin(); rpc_nodes_fuse_info != comm_edges_segments.end();
       ++rpc_nodes_fuse_info) {
    // Reorder the rpc node pairs list. Place monad inputs to the end of the list so that rpc send/recv nodes can be
    // built.
    std::vector<InterProcessOpPair> &inter_process_pairs = (*rpc_nodes_fuse_info).second;
    std::vector<InterProcessOpPair> monad_pairs;
    std::vector<InterProcessOpPair> no_monad_pairs;
    (void)std::for_each(inter_process_pairs.begin(), inter_process_pairs.end(), [&](const auto &op_pair) {
      if (HasAbstractMonad(std::get<1>(op_pair))) {
        (void)monad_pairs.emplace_back(op_pair);
      } else {
        (void)no_monad_pairs.emplace_back(op_pair);
      }
    });
    (void)no_monad_pairs.insert(no_monad_pairs.cend(), monad_pairs.cbegin(), monad_pairs.cend());
    inter_process_pairs = no_monad_pairs;

    std::vector<FusedInterProcessOpPair> fused_pairs;
    if (!common::GetEnv("fusion2").empty()) {
      fused_pairs = FuseCommEdges(inter_process_pairs);
    } else {
      std::vector<CNodePtr> rpc_send_nodes, rpc_recv_nodes;
      (void)std::for_each(inter_process_pairs.begin(), inter_process_pairs.end(),
                          [&rpc_send_nodes, &rpc_recv_nodes](const auto &node_pair) {
                            (void)rpc_send_nodes.emplace_back(std::get<0>(node_pair));
                            (void)rpc_recv_nodes.emplace_back(std::get<1>(node_pair));
                          });
      CNodePtr fused_send_node = FuseRpcSendNodes(rpc_send_nodes);
      CNodePtr fused_recv_node = FuseRpcRecvNodes(rpc_recv_nodes);

      for (size_t i = 0; i < inter_process_pairs.size(); i++) {
        FusedInterProcessOpPair fused_inter_process_pair =
          std::make_tuple(fused_send_node, fused_recv_node, i, std::get<2>(inter_process_pairs[i]),
                          std::get<3>(inter_process_pairs[i]));
        (void)fused_pairs.emplace_back(fused_inter_process_pair);
      }
    }
    results[rpc_nodes_fuse_info->first] = fused_pairs;
  }
  return results;
}

InterProcessOpEdgesInfo ParameterServerMode::FilterCommEdgesOfServerOptimizer(
  const InterProcessOpEdgesInfo &comm_edges) const {
  InterProcessOpEdgesInfo comm_edges_of_server_optimizer;
  for (const auto &edge_info : comm_edges) {
    if (edge_info.first.edge_label.label_name == kPSOptimizerEdgeLabel) {
      (void)comm_edges_of_server_optimizer.insert(edge_info);
    }
  }
  return comm_edges_of_server_optimizer;
}

FusedInterProcessOpPairMap ParameterServerMode::FilterNotServerOptimizerEdges(
  const InterProcessOpEdgesInfo &comm_edges) const {
  FusedInterProcessOpPairMap results;
  for (const auto &edge_info : comm_edges) {
    if (edge_info.first.edge_label.label_name != kPSOptimizerEdgeLabel) {
      const InterProcessOpEdge &edge = edge_info.first;
      const InterProcessOpPair &node_pair = edge_info.second;

      // We use the hash value to make these edges with index unique. So this index has no actual meaning.
      size_t edge_index = std::hash<std::string>{}(edge.to_string());
      InterProcessEdgeWithIndex edge_with_index = {edge.src_label, edge.dst_label, edge_index};
      FusedInterProcessOpPair fused_op_pair = std::make_tuple(std::get<0>(node_pair), std::get<1>(node_pair), 0,
                                                              std::get<2>(node_pair), std::get<3>(node_pair));
      std::vector<FusedInterProcessOpPair> pair_list = {fused_op_pair};
      (void)results.insert(std::make_pair(edge_with_index, pair_list));
    }
  }
  return results;
}

CNodePtr ParameterServerMode::FuseRpcSendNodes(const std::vector<CNodePtr> &rpc_send_nodes) {
  if (rpc_send_nodes.empty()) {
    MS_LOG(EXCEPTION) << "Rpc send node list is empty.";
  }
  std::vector<AnfNodePtr> send_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcSendOpName))};
  AbstractBasePtrList abstract_list;
  std::string fused_inter_process_edge_name = "";
  for (const auto &send_node : rpc_send_nodes) {
    MS_EXCEPTION_IF_NULL(send_node);
    for (size_t i = 1; i < send_node->inputs().size(); i++) {
      auto input_i = send_node->inputs()[i];
      MS_EXCEPTION_IF_NULL(input_i);
      // If the input of send is monad, do not pass it to fused send node.
      if (HasAbstractMonad(input_i)) {
        continue;
      }
      (void)send_inputs.emplace_back(input_i);
    }
    (void)abstract_list.emplace_back(send_node->abstract());
    (void)fused_inter_process_edge_name.append(
      common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(send_node, kAttrInterProcessEdgeNames).front());
  }

  CNodePtr fused_send_node = func_graph_->NewCNode(send_inputs);
  MS_EXCEPTION_IF_NULL(fused_send_node);
  fused_send_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  std::vector<std::string> fused_inter_process_edge_names = {fused_inter_process_edge_name};
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(fused_inter_process_edge_names), fused_send_node);
  common::AnfAlgo::CopyNodeAttr(kAttrPrimitiveTarget, rpc_send_nodes[0], fused_send_node);
  common::AnfAlgo::CopyNodeAttr(kAttrSendDstRanks, rpc_send_nodes[0], fused_send_node);
  common::AnfAlgo::CopyNodeAttr(kAttrSendDstRoles, rpc_send_nodes[0], fused_send_node);
  common::AnfAlgo::CopyNodeAttr(kAttrSendSrcNodeName, rpc_send_nodes[0], fused_send_node);
  common::AnfAlgo::CopyNodeAttr(kAttrSendDstNodeName, rpc_send_nodes[0], fused_send_node);
  return fused_send_node;
}

CNodePtr ParameterServerMode::FuseRpcRecvNodes(const std::vector<CNodePtr> &rpc_recv_nodes) {
  std::vector<AnfNodePtr> recv_inputs = {NewValueNode(std::make_shared<Primitive>(kRpcRecvOpName))};
  AbstractBasePtrList abstract_list;
  std::string fused_inter_process_edge_name = "";
  for (const auto &recv_node : rpc_recv_nodes) {
    MS_EXCEPTION_IF_NULL(recv_node);
    for (size_t i = 1; i < recv_node->inputs().size(); i++) {
      auto input_i = recv_node->inputs()[i];
      MS_EXCEPTION_IF_NULL(input_i);
      // If the input of recv is monad, do not pass it to fused recv node.
      if (HasAbstractMonad(input_i)) {
        continue;
      }
      (void)recv_inputs.emplace_back(input_i);
    }
    (void)abstract_list.emplace_back(recv_node->abstract());
    (void)fused_inter_process_edge_name.append(
      common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(recv_node, kAttrInterProcessEdgeNames).front());
  }
  // Add umonad for recv node to update reference.
  ValuePtr monad_value = kUMonad;
  auto monad_input = NewValueNode(monad_value);
  MS_EXCEPTION_IF_NULL(monad_input);
  monad_input->set_abstract(monad_value->ToAbstract());
  recv_inputs.push_back(monad_input);

  CNodePtr fused_recv_node = func_graph_->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(fused_recv_node);
  fused_recv_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  std::vector<std::string> fused_inter_process_edge_names = {fused_inter_process_edge_name};
  common::AnfAlgo::SetNodeAttr(kAttrInterProcessEdgeNames, MakeValue(fused_inter_process_edge_names), fused_recv_node);
  common::AnfAlgo::CopyNodeAttr(kAttrPrimitiveTarget, rpc_recv_nodes[0], fused_recv_node);
  common::AnfAlgo::CopyNodeAttr(kAttrRecvSrcRanks, rpc_recv_nodes[0], fused_recv_node);
  common::AnfAlgo::CopyNodeAttr(kAttrRecvSrcRoles, rpc_recv_nodes[0], fused_recv_node);
  common::AnfAlgo::CopyNodeAttr(kAttrRecvSrcNodeName, rpc_recv_nodes[0], fused_recv_node);
  common::AnfAlgo::CopyNodeAttr(kAttrRecvDstNodeName, rpc_recv_nodes[0], fused_recv_node);
  return fused_recv_node;
}

std::vector<FusedInterProcessOpPair> ParameterServerMode::FuseCommEdges(
  const std::vector<InterProcessOpPair> &inter_process_pairs) {
  std::vector<FusedInterProcessOpPair> fused_op_pairs;
  std::vector<CNodePtr> rpc_send_nodes, rpc_recv_nodes;
  std::map<size_t, size_t> indices_map;
  for (size_t i = 0; i < inter_process_pairs.size(); i++) {
    auto &op_pair = inter_process_pairs[i];
    auto reused_send_node =
      std::find_if(rpc_send_nodes.begin(), rpc_send_nodes.end(), [&op_pair](const auto &send_node_need_fuse) {
        CNodePtr send_node = std::get<0>(op_pair);
        auto node_name1 = common::AnfAlgo::GetInputNode(send_node, kIndex0)->fullname_with_scope();
        auto node_name2 = common::AnfAlgo::GetInputNode(send_node_need_fuse, kIndex0)->fullname_with_scope();
        return node_name1 == node_name2;
      });
    if (reused_send_node != rpc_send_nodes.end()) {
      size_t index = static_cast<size_t>(std::distance(rpc_send_nodes.begin(), reused_send_node));
      indices_map[i] = index;
    } else {
      (void)rpc_send_nodes.emplace_back(std::get<0>(op_pair));
      (void)rpc_recv_nodes.emplace_back(std::get<1>(op_pair));
      indices_map[i] = rpc_send_nodes.size() - 1;
    }
  }

  CNodePtr fused_send_node = FuseRpcSendNodes(rpc_send_nodes);
  CNodePtr fused_recv_node = FuseRpcRecvNodes(rpc_recv_nodes);
  for (size_t i = 0; i < inter_process_pairs.size(); i++) {
    FusedInterProcessOpPair fused_inter_process_pair =
      std::make_tuple(fused_send_node, fused_recv_node, indices_map[i], std::get<2>(inter_process_pairs[i]),
                      std::get<3>(inter_process_pairs[i]));
    (void)fused_op_pairs.emplace_back(fused_inter_process_pair);
  }
  return fused_op_pairs;
}

GraphSplitter::GraphSplitter(const FuncGraphPtr &func_graph, uint32_t rank_id, const std::string &role)
    : func_graph_(func_graph),
      rank_id_(rank_id),
      role_(role),
      exec_mode_(nullptr),
      this_process_label_({rank_id, role}),
      node_labels_{},
      need_fuse_rpc_nodes_(true) {
  // The distributed strategy is not explicitly defined by user. Distributed module generates the distributed strategy
  // and default label according to some flags set by other modules.
  mode_ = GenerateStrategy();
  default_label_ = {0, distributed::kEnvRoleOfWorker};
}

void EmbeddingCacheMode::PreBuildDistributedGraph() {
  // Only need add embedding cache ops of remote cache.
  if (role_ != distributed::kEnvRoleOfPServer) {
    return;
  }

  // 1. Add embedding cache ops of remote cache, and build service-side graph.
  AddEmbeddingCacheOps();

  // 2. Get node labels.
  MS_EXCEPTION_IF_NULL(node_labels_);
  node_labels_->clear();

  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  (void)std::for_each(all_nodes.begin(), all_nodes.end(), [this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      OperatorLabel label = GetNodeLabel(cnode);
      (void)node_labels_->emplace(node, label);
    }
  });
}

void EmbeddingCacheMode::AddEmbeddingCacheOps() const {
  uint32_t worker_num = ClusterContext::instance()->node_num(distributed::kEnvRoleOfWorker);
  if (worker_num == 0) {
    MS_LOG(EXCEPTION) << "In embedding cache mode, worker number should be greater than 0.";
  }

  // Build service-side graph.
  std::shared_ptr<parallel::PsEmbeddingCacheInserter> embedding_cache_inserter =
    std::make_shared<parallel::PsEmbeddingCacheInserter>(func_graph_, static_cast<int64_t>(rank_id_), role_,
                                                         worker_num);
  if (!embedding_cache_inserter->Run()) {
    MS_LOG(EXCEPTION) << "Insert ps embedding cache failed.";
  }
}

OperatorLabel EmbeddingCacheMode::GetNodeLabel(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only CNode has distributed split label.";
  }

  CNodePtr cnode = node->cast<CNodePtr>();
  auto prim_node = cnode->input(0);
  if (IsValueNode<Primitive>(prim_node)) {
    auto prim = GetValueNode<PrimitivePtr>(prim_node);
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->HasAttr(distributed::kOpLabelRankId) && prim->HasAttr(distributed::kOpLabelRole)) {
      MS_LOG(INFO) << "CNode which has distributed split label: " << cnode->fullname_with_scope();
      uint32_t rank_id = static_cast<uint32_t>(GetValue<int64_t>(prim->GetAttr(distributed::kOpLabelRankId)));
      std::string ms_role = GetValue<std::string>(prim->GetAttr(distributed::kOpLabelRole));
      return {rank_id, ms_role};
    }
  } else {
    // Get label for call node, 'call' node hasn't primitive to save attrs, so get attrs of 'call' from cnode.
    if (cnode->HasAttr(distributed::kOpLabelRankId) && cnode->HasAttr(distributed::kOpLabelRole)) {
      uint32_t rank_id = static_cast<uint32_t>(GetValue<int64_t>(cnode->GetAttr(distributed::kOpLabelRankId)));
      std::string ms_role = GetValue<std::string>(cnode->GetAttr(distributed::kOpLabelRole));
      return {rank_id, ms_role};
    }
  }
  return {rank_id_, role_};
}

GraphSplitter::~GraphSplitter() { node_labels_.clear(); }

void GraphSplitter::Run() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(func_graph_->manager());

  // Step 1: Dye all the nodes of the whole func_graph_.
  DyeGraph();
  // If all nodes are all on this process, no need to split the graph. So return.
  if (!NeedSplitGraph()) {
    MS_LOG(INFO) << "All nodes are on this precoess so there's no need to build and split distributed graph.";
    return;
  }

  // Step 2: Create exec_mode_ according to the current execution mode.
  CreateExecutionMode();

  // If this is general mode but no label is set, do not split graph to avoid unexpected optimizing out.
  if (mode_ == distributed::DistExecutionMode::kGeneralMode && !GraphHasLabel(func_graph_)) {
    MS_LOG(INFO) << "This graph has no label on it in general mode. So no need to split.";
    return;
  }

  // Step 3: Prebuild the distributed graph before it gets split.
  exec_mode_->PreBuildDistributedGraph();

  if (!NeedSplitGraph()) {
    MS_LOG(INFO) << "All nodes are on this precoess so there's no need to build and split distributed graph.";
    return;
  }

  // For TupleGetItem nodes, their label should be reset for good splitting performance.
  ReassignTupleGetItemNodeLabel();

  if (mode_ == distributed::DistExecutionMode::kGeneralMode) {
    // Only use ref sync mechanism when in general mode.
    ProcessRefNodes();
    // Add some control edges between different labels.
    AddExtraControlEdgeAcrossProcess();
  }

  // Step 4: Create inter-process operators for segments with different labels.
  InterProcessOpEdgesInfo comm_edges = GenerateInterProcessOperators();

  need_fuse_rpc_nodes_ = common::GetEnv(kEnvNeedFusion).empty() ? false : true;
  if (need_fuse_rpc_nodes_) {
    // Step 5: Fuse the rpc nodes to improve performance.
    const FusedInterProcessOpPairMap &fused_inter_process_op_pairs = exec_mode_->DoRpcNodeFusion(&comm_edges);

    // Step 6: Add dependency and eliminate extra nodes for fused rpc nodes.
    SplitGraph(fused_inter_process_op_pairs);

    // Step 7: Postbuild the graph after splitting with fused edges.
    exec_mode_->PostBuildDistributedGraph(fused_inter_process_op_pairs);
  } else {
    // Step 5: Generate the node segments with different labels.
    std::vector<SplitGraphSegment> segments = GenerateSplitSegments();
    // If the segment number is 0, there will be no distributed execution.
    if (segments.empty()) {
      return;
    }

    // Step 6: Split the graph and eliminate extra nodes.
    SplitGraph(segments, comm_edges);

    // Step 7: Postbuild the graph after splitting.
    exec_mode_->PostBuildDistributedGraph(comm_edges);
  }
  // Only eliminate the data-sync node pairs in general mode.
  if (mode_ == distributed::DistExecutionMode::kGeneralMode) {
    EliminateDataSyncNode();
    EliminateControlEdgeNode();
  }
}

void GraphSplitter::DyeGraph() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  (void)std::for_each(all_nodes.begin(), all_nodes.end(), [this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    // Mark all nodes with original label at the beginning. This means the node is supposed to be on the process with
    // default_label_.
    node_labels_[node] = default_label_;
    if (node->isa<CNode>()) {
      // For CNodes, mark them with the label passed by frontend if has one.
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      OperatorLabel label = GetSplitLabel(cnode);
      node_labels_[node] = label;
    }

    // If the node's label is the same as this process's, set its label to this_process_label_.
    if (this_process_label_.LooseEqual(node_labels_[node], mode_)) {
      node_labels_[node] = this_process_label_;
    }
  });
}

void GraphSplitter::CreateExecutionMode() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (node_labels_.empty()) {
    MS_LOG(EXCEPTION) << "Must dye the original graph before creating execution mode.";
  }
  if (mode_ == distributed::DistExecutionMode::kPSMode) {
    exec_mode_ = std::make_unique<ParameterServerMode>(func_graph_, &node_labels_, rank_id_, role_);
  } else if (mode_ == distributed::DistExecutionMode::kEmbeddingCacheMode) {
    exec_mode_ = std::make_unique<EmbeddingCacheMode>(func_graph_, &node_labels_, rank_id_, role_);
  } else if (mode_ == distributed::DistExecutionMode::kParallelMode) {
    exec_mode_ = std::make_unique<ParallelMode>(func_graph_, &node_labels_, rank_id_, role_);
  } else if (mode_ == distributed::DistExecutionMode::kGeneralMode) {
    exec_mode_ = std::make_unique<GeneralMode>(func_graph_, &node_labels_, rank_id_, role_);
  }
  MS_EXCEPTION_IF_NULL(exec_mode_);
}

std::vector<SplitGraphSegment> GraphSplitter::GenerateSplitSegments() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  auto return_node = func_graph_->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> nodes = FuncGraph::TopoSort(return_node);

  std::vector<SplitGraphSegment> results;
  SplitGraphSegment segment;
  OperatorLabel last_label = this_process_label_;
  segment.label = last_label;
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto cnode_split_label = node_labels_[n];
    // If this node's label is not the same as last node's, create a segment from 'segment_nodes'.
    if (cnode_split_label != last_label && !segment.nodes.empty()) {
      (void)results.emplace_back(segment);
      segment.nodes.clear();
    }
    // Mark the last label.
    last_label = cnode_split_label;
    segment.label = cnode_split_label;
    (void)segment.nodes.emplace_back(n);
  }

  // Add the last segment.
  (void)results.emplace_back(segment);
  MS_LOG(INFO) << "Segments number with different distributed split labels is " << results.size();
  return results;
}

void GraphSplitter::ReassignTupleGetItemNodeLabel() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  AnfNodePtrList all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  for (const auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      node_labels_[node] = RecursiveSetTupeGetItemLabel(node->cast<CNodePtr>());
    }
  }
}

OperatorLabel GraphSplitter::RecursiveSetTupeGetItemLabel(const CNodePtr &tuple_get_item_node) {
  // Return if this node has already been visited.
  if (visited_tuple_get_item_nodes_.count(tuple_get_item_node) != 0) {
    if (NodeHasLabel(tuple_get_item_node)) {
      return node_labels_[tuple_get_item_node];
    } else {
      MS_LOG(EXCEPTION) << "TupeGetItem node " << tuple_get_item_node->fullname_with_scope() << " has no lebel.";
    }
  }

  visited_tuple_get_item_nodes_[tuple_get_item_node] = true;
  auto tuple_input = common::AnfAlgo::GetInputNode(tuple_get_item_node, kIndex0);
  OperatorLabel tuple_get_item_label;
  if (IsPrimitiveCNode(tuple_input, prim::kPrimTupleGetItem)) {
    // If TupleGetItem's input is a TupleGetItem node, recursively trace up and get a proper input's label.
    tuple_get_item_label = RecursiveSetTupeGetItemLabel(tuple_input->cast<CNodePtr>());
    node_labels_[tuple_input] = tuple_get_item_label;
  } else {
    // Set TupleGetItem's label the same as its input so it's easier to split.
    tuple_get_item_label = node_labels_[tuple_input];
  }
  return tuple_get_item_label;
}

void GraphSplitter::ProcessRefNodes() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  AnfNodePtrList all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  // Traverse all nodes and find each nodes with side effect.
  CNodePtrList cnodes_with_side_effect = GetSideEffectNodeList(all_nodes);
  for (const auto &cnode : cnodes_with_side_effect) {
    // Filter out all ref inputs which need to be synchronized between different processes.
    AnfNodePtrList ref_inputs = GetRefInputs(cnode);
    // Get the user node(UpdateState) of side effect node.
    CNodePtr update_state_node = FindNextUpdateStateNode(func_graph_, cnode);
    MS_EXCEPTION_IF_NULL(update_state_node);

    // The key method to keep the correctness of reference nodes across computing graph nodes.
    AddDataSyncNode(cnode, update_state_node, ref_inputs);
  }
}

void GraphSplitter::AddExtraControlEdgeAcrossProcess() { AddControlEdgeForProcessWithoutIndegree(); }

InterProcessOpEdgesInfo GraphSplitter::GenerateInterProcessOperators() {
  InterProcessOpEdgesInfo comm_edges;
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    // Only support to split CNode to other process.
    if (!node->isa<CNode>()) {
      continue;
    }

    // Generating send/recv nodes for each nodes' inputs will be enough.
    auto node_inputs_comm_edges = GenerateInterProcessOpsForNodeInputs(node);
    comm_edges.insert(node_inputs_comm_edges.cbegin(), node_inputs_comm_edges.cend());
  }
  MS_LOG(INFO) << "The communication edge number is " << comm_edges.size();
  return comm_edges;
}

void GraphSplitter::SplitGraph(const std::vector<SplitGraphSegment> &segments,
                               const InterProcessOpEdgesInfo &comm_edges) {
  // Step 1: Traverse all the segments to add Depend for this process's graph.
  // The list of corresponding in and out degrees. In another word, the map between one segments' input send nodes. and
  // output recv nodes.
  InOutDegreeList in_out_degree_list = GenerateInOutDegreeList(segments, comm_edges);
  if (in_out_degree_list.empty()) {
    MS_LOG(WARNING) << "After splitting, this process has no graph on it. So optimize out the whole graph.";
    auto return_value_node = CreateReplacedOutputNode(func_graph_, func_graph_->output());
    (void)func_graph_->manager()->Replace(func_graph_->output(), return_value_node);
    return;
  }

  // Step 2: Add dependency between communication edges on this process.
  AddDependencyBetweenEdges(comm_edges);

  // Step 3: Eliminate nodes not on this process.
  EliminateExtraNodes(comm_edges);
}

void GraphSplitter::SplitGraph(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs) {
  if (fused_inter_process_op_pairs.empty()) {
    MS_LOG(WARNING) << "After splitting, this process has no graph on it. So optimize out the whole graph.";
    auto return_value_node = CreateReplacedOutputNode(func_graph_, func_graph_->output());
    (void)func_graph_->manager()->Replace(func_graph_->output(), return_value_node);
    return;
  }

  // Step 1: Replace origin nodes with recv nodes.
  ReplaceOriginNodesWithRecv(fused_inter_process_op_pairs);

  // Step 2: Connect output for send nodes.
  AddDependencyForSend(fused_inter_process_op_pairs);
}

void GraphSplitter::AddDataSyncNode(const CNodePtr &side_effect_node, const CNodePtr &update_state_node,
                                    const AnfNodePtrList &ref_nodes) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(side_effect_node);
  MS_EXCEPTION_IF_NULL(update_state_node);

  MS_EXCEPTION_IF_CHECK_FAIL(
    (node_labels_.count(side_effect_node) != 0),
    "The node label for side effect node " + side_effect_node->fullname_with_scope() + " is not set.");
  auto side_effect_node_label = node_labels_[side_effect_node];

  for (const auto &ref : ref_nodes) {
    std::set<OperatorLabel> diff_labels;
    for (const auto &user : func_graph_->manager()->node_users()[ref]) {
      const auto &user_node = user.first;
      MS_LOG(DEBUG) << "The user of ref " << ref->fullname_with_scope() << " is " << user_node->fullname_with_scope()
                    << ", side-effect node label: " << side_effect_node_label.to_string()
                    << ", user node label: " << node_labels_[user_node].to_string();
      if (node_labels_[user_node] != side_effect_node_label) {
        diff_labels.insert(node_labels_[user_node]);
      } else {
        // If the user node is Load, we need to find one next user of it so the node could be correctly split.
        if (IsPrimitiveCNode(user_node, prim::kPrimLoad)) {
          for (const auto &load_user : func_graph_->manager()->node_users()[user_node]) {
            const auto &load_user_node = load_user.first;
            MS_LOG(DEBUG) << "Load user is " << load_user_node
                          << ", label: " << node_labels_[load_user_node].to_string();
            if (node_labels_[load_user_node] != side_effect_node_label) {
              diff_labels.insert(node_labels_[load_user_node]);
            }
          }
        }
      }
    }
    // If the ref is used in multiple compute graph nodes, it needs to be synchronized.
    if (diff_labels.empty()) {
      MS_LOG(INFO) << "No need to synchronize ref node " << ref->fullname_with_scope()
                   << " because the user nodes are on the same process.";
      continue;
    }

    //  Create data-sync nodes and connect them to UpdateState node.
    auto data_sync_node_list = CreateDataSyncNodes(side_effect_node, ref, diff_labels);
    for (const auto &node_pair : data_sync_node_list) {
      CNodePtr src_node = node_pair.first;
      CNodePtr dst_node = node_pair.second;
      func_graph_->manager()->AddEdge(update_state_node, dst_node);
    }
  }
}

DataSyncNodePairList GraphSplitter::CreateDataSyncNodes(const CNodePtr &side_effect_node, const AnfNodePtr &ref,
                                                        const std::set<OperatorLabel> &diff_labels) {
  MS_EXCEPTION_IF_NULL(side_effect_node);
  MS_EXCEPTION_IF_NULL(ref);

  DataSyncNodePairList result;
  for (const auto &label : diff_labels) {
    // Data sync src node.
    std::vector<AnfNodePtr> sync_src_node_inputs = {
      NewValueNode(std::make_shared<Primitive>(distributed::kDataSyncSrcOpName))};
    sync_src_node_inputs.emplace_back(ref);
    sync_src_node_inputs.emplace_back(side_effect_node);
    CNodePtr sync_src_node = func_graph_->NewCNode(sync_src_node_inputs);
    MS_EXCEPTION_IF_NULL(sync_src_node);
    sync_src_node->set_abstract(ref->abstract());
    node_labels_[sync_src_node] = node_labels_[side_effect_node];

    // Data sync dst node.
    std::vector<AnfNodePtr> sync_dst_node_inputs = {
      NewValueNode(std::make_shared<Primitive>(distributed::kDataSyncDstOpName))};
    sync_dst_node_inputs.emplace_back(sync_src_node);
    CNodePtr sync_dst_node = func_graph_->NewCNode(sync_dst_node_inputs);
    MS_EXCEPTION_IF_NULL(sync_dst_node);
    auto fake_value = CreateFakeValueNode(false);
    MS_EXCEPTION_IF_NULL(fake_value);
    sync_dst_node->set_abstract(fake_value->abstract());
    node_labels_[sync_dst_node] = label;

    MS_LOG(INFO) << "Data sync pair: " << sync_src_node->fullname_with_scope() << "_"
                 << node_labels_[sync_src_node].to_string() << "->" << sync_dst_node->fullname_with_scope() << "_"
                 << label.to_string();
    result.push_back(std::make_pair(sync_src_node, sync_dst_node));
  }
  return result;
}

void GraphSplitter::AddControlEdgeForProcessWithoutIndegree() {
  std::for_each(node_labels_.begin(), node_labels_.end(),
                [this](const auto &node_label_pair) { all_labels_.insert(node_label_pair.second); });

  std::set<OperatorLabel> labels_has_indegree;
  AnfNodePtrList all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = kIndex1; i < cnode->size(); ++i) {
      const auto &input = cnode->inputs().at(i);
      if (NodeHasLabel(input) && NodeHasLabel(cnode) && node_labels_[input] != node_labels_[cnode] &&
          input->isa<CNode>()) {
        MS_LOG(DEBUG) << "Label " << node_labels_[cnode].to_string() << " has indegree from label "
                      << node_labels_[input].to_string() << ", edge: " << input->fullname_with_scope() << " to "
                      << cnode->fullname_with_scope();
        labels_has_indegree.insert(node_labels_[cnode]);
      }
    }
  }

  ControlEdgeNodePairList control_edge_node_pair_list;
  for (const OperatorLabel &label : all_labels_) {
    // If this label has no indegree, add extra control edge nodes.
    if (labels_has_indegree.count(label) == 0) {
      ControlEdgeNodePair control_edge_nodes = CreateControlEdgeNode(default_label_, label);
      control_edge_node_pair_list.emplace_back(control_edge_nodes);
    }
  }

  if (!control_edge_node_pair_list.empty()) {
    // Connect the dangling control dst nodes to the output.
    AnfNodePtrList make_tuple_inputs;
    std::for_each(control_edge_node_pair_list.begin(), control_edge_node_pair_list.end(),
                  [&make_tuple_inputs](const auto &node_pair) {
                    CNodePtr control_dst_node = node_pair.second;
                    make_tuple_inputs.emplace_back(control_dst_node);
                  });

    // Make tuple for all control-edge dst nodes.
    MS_EXCEPTION_IF_NULL(func_graph_);
    auto tuple_of_control_dst_nodes = CreateMakeTupleNode(func_graph_, make_tuple_inputs);
    MS_EXCEPTION_IF_NULL(tuple_of_control_dst_nodes);
    node_labels_[tuple_of_control_dst_nodes] = default_label_;

    // Add dependency to the Return node so control-edge nodes won't be optimized out.
    AnfNodePtrList depend_inputs = {NewValueNode(prim::kPrimDepend), func_graph_->output(), tuple_of_control_dst_nodes};
    auto final_output_node = func_graph_->NewCNode(depend_inputs);
    MS_EXCEPTION_IF_NULL(final_output_node);
    node_labels_[final_output_node] = default_label_;

    final_output_node->set_abstract(func_graph_->output()->abstract());
    (void)func_graph_->manager()->SetEdge(func_graph_->get_return(), kIndex1, final_output_node);
  }
  return;
}

ControlEdgeNodePair GraphSplitter::CreateControlEdgeNode(const OperatorLabel &src_label,
                                                         const OperatorLabel &dst_label) {
  // Control src node's input is a value node. It has not practical meaning.
  auto fake_tensor = std::make_shared<tensor::Tensor>(1.0);
  MS_EXCEPTION_IF_NULL(fake_tensor);
  auto fake_value = NewValueNode(fake_tensor);
  MS_EXCEPTION_IF_NULL(fake_value);
  fake_value->set_abstract(fake_tensor->ToAbstract());

  AnfNodePtrList control_src_inputs = {NewValueNode(std::make_shared<Primitive>(distributed::kControlSrcOpName)),
                                       fake_value};
  CNodePtr control_src_node = func_graph_->NewCNode(control_src_inputs);
  MS_EXCEPTION_IF_NULL(control_src_node);
  control_src_node->set_abstract(fake_value->abstract());
  node_labels_[control_src_node] = src_label;

  // Control dst node's input is control src node.
  AnfNodePtrList control_dst_inputs = {NewValueNode(std::make_shared<Primitive>(distributed::kControlDstOpName)),
                                       control_src_node};
  CNodePtr control_dst_node = func_graph_->NewCNode(control_dst_inputs);
  MS_EXCEPTION_IF_NULL(control_dst_node);
  control_dst_node->set_abstract(control_src_node->abstract());
  node_labels_[control_dst_node] = dst_label;

  // At this phase, the control_dst_node is still a dangling node. We need to connect it to the output to avoid
  // optimizing out.
  return std::make_pair(control_src_node, control_dst_node);
}

void GraphSplitter::EliminateDataSyncNode() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  AnfNodePtrList all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (common::AnfAlgo::GetCNodeName(cnode) == distributed::kDataSyncSrcOpName) {
      if (cnode->inputs().size() != kSizeThree) {
        MS_LOG(EXCEPTION) << "Node DataSyncSrc's input number should be 3, but got " << cnode->inputs().size();
      }
      // The first input is parameter and the second input is side effect node.
      auto param_node = cnode->inputs()[kIndex1];
      MS_EXCEPTION_IF_NULL(param_node);
      auto side_effect_node = cnode->inputs()[kIndex2];
      MS_EXCEPTION_IF_NULL(side_effect_node);
      MS_LOG(DEBUG) << "Parameter node is " << param_node->fullname_with_scope() << ", side effect node is "
                    << side_effect_node->fullname_with_scope();

      AnfNodePtrList update_state_inputs = {side_effect_node};
      CNodePtr update_state_node = CreateUpdateStateNode(func_graph_, update_state_inputs);
      MS_EXCEPTION_IF_NULL(update_state_node);

      // For parameter, connect it to a 'Load' node so that the control arrow could be correctly linked.
      AnfNodePtrList load_inputs = {NewValueNode(prim::kPrimLoad), param_node, update_state_node};

      auto load_node_replace_data_sync_src = func_graph_->NewCNode(load_inputs);
      MS_EXCEPTION_IF_NULL(load_node_replace_data_sync_src);
      load_node_replace_data_sync_src->set_abstract(cnode->abstract());
      func_graph_->manager()->Replace(cnode, load_node_replace_data_sync_src);
    } else if (common::AnfAlgo::GetCNodeName(cnode) == distributed::kDataSyncDstOpName) {
      if (cnode->inputs().size() != kSizeTwo) {
        MS_LOG(EXCEPTION) << "Node DataSyncDst's input number should be 2, but got " << cnode->inputs().size();
      }
      auto input_node = cnode->inputs()[kIndex1];
      MS_EXCEPTION_IF_NULL(input_node);

      auto users = func_graph_->manager()->node_users()[cnode];
      for (const auto &user_pair : users) {
        auto user_node = user_pair.first;
        int input_index = user_pair.second;
        func_graph_->manager()->SetEdge(user_node, input_index, input_node);
      }
    }
  }
}

void GraphSplitter::EliminateControlEdgeNode() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  AnfNodePtrList all_nodes = DeepScopedGraphSearch(func_graph_->get_return());
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (common::AnfAlgo::GetCNodeName(cnode) == distributed::kControlSrcOpName) {
      // ControlSrc->RpcSend is converted to FakeValue->RpcSend.
      auto fake_value_node = CreateFakeValueNode(false);
      MS_EXCEPTION_IF_NULL(fake_value_node);
      (void)func_graph_->manager()->Replace(cnode, fake_value_node);
    } else if (common::AnfAlgo::GetCNodeName(cnode) == distributed::kControlDstOpName) {
      if (cnode->inputs().size() != kSizeTwo) {
        MS_LOG(EXCEPTION) << "Node DataSyncDst's input number should be 2, but got " << cnode->inputs().size();
      }
      auto input_node = cnode->inputs()[kIndex1];
      MS_EXCEPTION_IF_NULL(input_node);
      (void)func_graph_->manager()->Replace(cnode, input_node);
    }
  }
}

void GraphSplitter::DumpDistributedGraph(const InterProcessOpEdgesInfo &comm_edges) {
  // Traverse all the segments to add Depend for this process's graph.
  for (const auto &edge : comm_edges) {
    auto send_recv_pair = edge.second;
    auto send_node = std::get<0>(send_recv_pair);
    auto recv_node = std::get<1>(send_recv_pair);
    auto user_node = std::get<2>(send_recv_pair);
    auto user_node_index = std::get<3>(send_recv_pair);
    func_graph_->manager()->SetEdge(recv_node, 1, send_node);
    func_graph_->manager()->SetEdge(user_node, user_node_index, recv_node);
  }
  MS_LOG(INFO) << "Cut graph without eliminating nodes.";
  draw::Draw("single_node_graph.dot", func_graph_);
}

OperatorLabel GraphSplitter::GetSplitLabel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only CNode has distributed split label.";
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim_node = cnode->input(0);
  if (IsValueNode<Primitive>(prim_node)) {
    TransformPrimAttrToAttr(cnode);
    auto prim = GetValueNode<PrimitivePtr>(prim_node);
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->HasAttr(distributed::kOpLabelRankId) && prim->HasAttr(distributed::kOpLabelRole)) {
      MS_LOG(INFO) << "CNode which has distributed split label: " << cnode->fullname_with_scope();
      uint32_t rank_id = static_cast<uint32_t>(GetValue<int64_t>(prim->GetAttr(distributed::kOpLabelRankId)));
      std::string ms_role = GetValue<std::string>(prim->GetAttr(distributed::kOpLabelRole));
      return {rank_id, ms_role};
    }
  } else {
    // Get label for call node, 'call' node hasn't primitive to save attrs, so get attrs of 'call' from cnode.
    if (cnode->HasAttr(distributed::kOpLabelRankId) && cnode->HasAttr(distributed::kOpLabelRole)) {
      uint32_t rank_id = static_cast<uint32_t>(GetValue<int64_t>(cnode->GetAttr(distributed::kOpLabelRankId)));
      std::string ms_role = GetValue<std::string>(cnode->GetAttr(distributed::kOpLabelRole));
      return {rank_id, ms_role};
    }
  }
  return default_label_;
}

InterProcessOpEdgesInfo GraphSplitter::GenerateInterProcessOpsForNodeInputs(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  InterProcessOpEdgesInfo comm_edges;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_i = cnode->inputs()[i];
    MS_EXCEPTION_IF_NULL(input_i);

    // If the input's not a cnode, or its label is the same as this node's, or the input is 'Load' node for parameter,
    // there's no need to add communication nodes.
    if (!input_i->isa<CNode>() || IsNodesWithSameLabel(input_i, cnode) ||
        common::AnfAlgo::GetCNodeName(input_i) == "Load") {
      if (IsOneOfRealGraphInput(func_graph_, input_i) && !IsNodesWithSameLabel(input_i, cnode)) {
        MS_LOG(INFO) << "The input " << input_i->fullname_with_scope() << " needs to be split.";
      } else {
        continue;
      }
    }

    InterProcessEdgeLabel edge_label = GenerateEdgeLabel(input_i, cnode);
    InterProcessOpEdge edge = {input_i, node_labels_[input_i], cnode, node_labels_[cnode], edge_label};

    auto send_node = CreateSendNode(func_graph_, edge);
    MS_EXCEPTION_IF_NULL(send_node);
    // The label should be the same as the node which will 'launch' Send node.
    node_labels_[send_node] = edge.src_label;

    auto recv_node = CreateRecvNode(func_graph_, edge);
    MS_EXCEPTION_IF_NULL(recv_node);
    // The label should be the same as the node which Receives the 'input'.
    node_labels_[recv_node] = edge.dst_label;

    auto comm_node_pair = std::make_tuple(send_node, recv_node, cnode, SizeToInt(i));
    (void)comm_edges.insert(std::make_pair(edge, comm_node_pair));
  }
  return comm_edges;
}

InterProcessEdgeLabel GraphSplitter::GenerateEdgeLabel(const AnfNodePtr &src_node, const AnfNodePtr &dst_node) const {
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);
  std::string src_node_edge_label = "";
  std::string dst_node_edge_label = "";
  if (src_node->isa<CNode>()) {
    src_node_edge_label = common::AnfAlgo::HasNodeAttr(kAttrInterProcessEdgeLabel, src_node->cast<CNodePtr>())
                            ? common::AnfAlgo::GetNodeAttr<std::string>(src_node, kAttrInterProcessEdgeLabel)
                            : "";
  }
  if (dst_node->isa<CNode>()) {
    dst_node_edge_label = common::AnfAlgo::HasNodeAttr(kAttrInterProcessEdgeLabel, dst_node->cast<CNodePtr>())
                            ? common::AnfAlgo::GetNodeAttr<std::string>(dst_node, kAttrInterProcessEdgeLabel)
                            : "";
  }
  if (!src_node_edge_label.empty() && !dst_node_edge_label.empty()) {
    if (src_node_edge_label != dst_node_edge_label) {
      MS_LOG(EXCEPTION) << "The edge label name of src node and dst node should be same."
                        << src_node->fullname_with_scope() << "->" << dst_node->fullname_with_scope();
    }
  }
  InterProcessEdgeLabel edge_label;
  if (!src_node_edge_label.empty()) {
    edge_label.label_name = src_node_edge_label;
  } else if (!dst_node_edge_label.empty()) {
    edge_label.label_name = dst_node_edge_label;
  } else {
    MS_LOG(DEBUG) << "Edge label is empty for " << src_node->fullname_with_scope() << "->"
                  << dst_node->fullname_with_scope();
  }
  return edge_label;
}

std::vector<AnfNodePtr> GraphSplitter::FindInterProcessInDegree(const std::vector<AnfNodePtr> &nodes,
                                                                const InterProcessOpEdgesInfo &comm_edges) {
  std::vector<AnfNodePtr> results;
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }

    CNodePtr cnode = n->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      auto input_i = cnode->inputs()[i];
      InterProcessOpEdge edge = {input_i, node_labels_[input_i], cnode, node_labels_[cnode]};
      if (comm_edges.count(edge) != 0 && edge.src_label == this_process_label_) {
        MS_LOG(INFO) << edge.to_string() << " is a communication edge.";
        auto comm_node_pair = comm_edges.at(edge);
        (void)results.emplace_back(std::get<0>(comm_node_pair));
      }
    }
  }
  return results;
}

std::vector<AnfNodePtr> GraphSplitter::FindInterProcessOutDegree(const std::vector<AnfNodePtr> &nodes,
                                                                 const InterProcessOpEdgesInfo &comm_edges) {
  std::vector<AnfNodePtr> results;
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }

    CNodePtr cnode = n->cast<CNodePtr>();
    auto users = func_graph_->manager()->node_users()[cnode];
    for (auto &u : users) {
      auto user_node = u.first->cast<CNodePtr>();
      InterProcessOpEdge edge = {cnode, node_labels_[cnode], user_node, node_labels_[user_node]};
      if (comm_edges.count(edge) != 0 && edge.dst_label == this_process_label_) {
        MS_LOG(INFO) << edge.to_string() << " is a communication edge.";
        auto comm_node_pair = comm_edges.at(edge);
        (void)results.emplace_back(std::get<1>(comm_node_pair));
      }
    }
  }
  return results;
}

InOutDegreeList GraphSplitter::GenerateInOutDegreeList(const std::vector<SplitGraphSegment> &segments,
                                                       const InterProcessOpEdgesInfo &comm_edges) {
  MS_LOG(INFO) << "Start finding inter-process in-degrees.";

  InOutDegreeList in_out_degree_list;
  // Traverse all the segments to add Depend for this process's graph.
  for (const auto &segment : segments) {
    // If this segment should be on current process, continue.
    if (segment.label == this_process_label_) {
      continue;
    }
    std::vector<AnfNodePtr> nodes = segment.nodes;
    if (nodes.empty()) {
      MS_LOG(EXCEPTION) << "This segment is empty.";
      return in_out_degree_list;
    }

    auto segment_first_node = nodes[0];
    if (node_labels_[segment_first_node] != segment.label) {
      MS_LOG(EXCEPTION) << "Node label " << node_labels_[segment_first_node].to_string()
                        << " is not the same as segment label " << segment.label.to_string();
    }

    // Prepare for adding Depend between in-degree and out-degree of this segment because the execution order should
    // be kept consistent.
    std::vector<AnfNodePtr> concerned_in_degree_nodes = FindInterProcessInDegree(nodes, comm_edges);
    std::vector<AnfNodePtr> concerned_out_degree_nodes = FindInterProcessOutDegree(nodes, comm_edges);
    if (!concerned_in_degree_nodes.empty() || !concerned_out_degree_nodes.empty()) {
      (void)in_out_degree_list.emplace_back(std::make_pair(concerned_in_degree_nodes, concerned_out_degree_nodes));
    }
  }
  MS_LOG(INFO) << "End finding inter-process in-degrees.";
  return in_out_degree_list;
}

void GraphSplitter::AddDependencyBetweenEdges(const InterProcessOpEdgesInfo &comm_edges) {
  // 'in_degree_comm_edges' is the edges with recv node on this process.
  InterProcessOpEdgesInfo in_degree_comm_edges;
  // 'out_degree_comm_edges' is the edges with send node on this process.
  InterProcessOpEdgesInfo out_degree_comm_edges;

  // Src nodes of RpcSend nodes.
  AnfNodePtrSet send_src_nodes;
  // Map of src nodes to its all RpcSend nodes.
  std::map<AnfNodePtr, AnfNodePtrSet> src_nodes_to_send_nodes;
  // This map represents which send nodes are hung. Key is RpcSend node.
  std::map<AnfNodePtr, bool> is_send_node_hung;
  for (const auto &e : comm_edges) {
    const InterProcessOpEdge &edge_info = e.first;
    const InterProcessOpPair &op_pair = e.second;

    if (edge_info.src_label == this_process_label_) {
      const AnfNodePtr &send_src_node = edge_info.src_node;
      const AnfNodePtr &rpc_send_node = std::get<0>(op_pair);
      send_src_nodes.insert(send_src_node);
      src_nodes_to_send_nodes[send_src_node].insert(rpc_send_node);
      is_send_node_hung[rpc_send_node] = true;

      MS_LOG(DEBUG) << "Out degree edge: " << edge_info.to_string() << ". Send src node "
                    << send_src_node->fullname_with_scope() << " has RpcSend node "
                    << rpc_send_node->fullname_with_scope();
      out_degree_comm_edges[edge_info] = op_pair;
    }

    if (edge_info.dst_label == this_process_label_) {
      MS_LOG(DEBUG) << "In degree edge: " << edge_info.to_string();
      in_degree_comm_edges[edge_info] = op_pair;
    }
  }

  // This step is vital. It builds a map consists of all dependencies to send src nodes, which helps to
  // add explicit dependency edges for RpcSend and RpcRecv nodes.
  std::map<AnfNodePtr, AnfNodePtrSet> node_dependency = FilterDependencyToTargetNode(func_graph_, send_src_nodes);
  MS_LOG(INFO) << "After filtering out the dependencies, add dependency edges between RpcSend and RpcRecv nodes.";

  // Connect RpcSend and RpcRecv with minimal dependencies.
  AddSendRecvDependency(in_degree_comm_edges, send_src_nodes, src_nodes_to_send_nodes, node_dependency,
                        &is_send_node_hung);

  // Some RpcSend nodes may be hung, we need to connect these nodes to output in case they are optimized out.
  AnfNodePtrList hung_nodes_list;
  for (const auto &is_hung : is_send_node_hung) {
    if (is_hung.second) {
      MS_LOG(INFO) << "RpcSend node: " << is_hung.first->fullname_with_scope() << " is hung.";
      hung_nodes_list.emplace_back(is_hung.first);
    }
  }
  if (!hung_nodes_list.empty()) {
    HandleHungNodes(func_graph_, node_labels_, this_process_label_, hung_nodes_list);
  }
}

void GraphSplitter::AddDependencyBetweenSegments(const InOutDegreeList &in_out_degree_list) {
  MS_LOG(INFO) << "Start adding dependency between segments.";
  // This tuple is key to the dependency of send nodes so that they will not be optimized out in some cases.
  std::vector<AnfNodePtr> send_node_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < in_out_degree_list.size(); i++) {
    auto &concerned_in_degree_nodes = in_out_degree_list[i].first;
    auto &concerned_out_degree_nodes = in_out_degree_list[i].second;
    (void)send_node_tuple_inputs.insert(send_node_tuple_inputs.cend(), concerned_in_degree_nodes.cbegin(),
                                        concerned_in_degree_nodes.cend());
    if (concerned_out_degree_nodes.empty()) {
      // If this is the last segment's in and out degrees and has no out degrees, connect the send nodes to graph's
      // output.
      if (i == in_out_degree_list.size() - 1) {
        auto make_tuple_node = func_graph_->NewCNode(send_node_tuple_inputs);
        AbstractBasePtrList abstract_list;
        (void)std::for_each(send_node_tuple_inputs.cbegin() + 1, send_node_tuple_inputs.cend(),
                            [&](const auto &input) { (void)abstract_list.emplace_back(input->abstract()); });
        make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));

        // Connect fused send nodes to the output so they will not be optimized out.
        AnfNodePtr origin_output = func_graph_->output();
        if (node_labels_.count(origin_output) == 0) {
          MS_LOG(EXCEPTION) << "The origin output node " << origin_output->fullname_with_scope()
                            << " should have corresponding operator label.";
        }

        // If the output is not on this process, replace it with a fake output node.
        AnfNodePtr replaced_output = nullptr;
        if (node_labels_[origin_output] != this_process_label_) {
          replaced_output = CreateReplacedOutputNode(func_graph_, origin_output);
        } else {
          replaced_output = origin_output;
        }

        // Add dependency and replace.
        std::vector<AnfNodePtr> depend_inputs = {NewValueNode(prim::kPrimDepend), replaced_output, make_tuple_node};
        auto final_output_node = func_graph_->NewCNode(depend_inputs);
        MS_EXCEPTION_IF_NULL(final_output_node);
        final_output_node->set_abstract(replaced_output->abstract());
        (void)func_graph_->manager()->SetEdge(func_graph_->get_return(), 1, final_output_node);
      }
    } else {
      auto make_tuple_node = func_graph_->NewCNode(send_node_tuple_inputs);
      for (auto &recv : concerned_out_degree_nodes) {
        std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), recv->cast<CNodePtr>()->inputs()[1],
                                                make_tuple_node};
        auto depend = func_graph_->NewCNode(depend_input);
        depend->set_abstract(recv->cast<CNodePtr>()->inputs()[1]->abstract());
        func_graph_->manager()->SetEdge(recv, 1, depend);
      }
      // Reset the make tuple node inputs for next segments in degrees.
      send_node_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    }
  }
  MS_LOG(INFO) << "End adding dependency between segments.";
}

void GraphSplitter::EliminateExtraNodes(const InterProcessOpEdgesInfo &comm_edges) {
  MS_LOG(INFO) << "Start eliminating nodes not on this process.";
  // Eliminate nodes which should be launched by other processes by set output edge.
  for (auto &edge : comm_edges) {
    InterProcessOpPair send_recv_pair = edge.second;
    auto send_node = std::get<0>(send_recv_pair);
    auto recv_node = std::get<1>(send_recv_pair);
    auto user_node = std::get<2>(send_recv_pair);
    int user_node_index = std::get<3>(send_recv_pair);

    OperatorLabel send_label = node_labels_[send_node];
    OperatorLabel recv_label = node_labels_[recv_node];
    if (send_label == recv_label) {
      MS_LOG(EXCEPTION) << "The Send and Recv must have different label. But got Send: " << send_label.to_string()
                        << ", Recv: " << recv_label.to_string();
    }

    if (recv_label == this_process_label_) {
      func_graph_->manager()->SetEdge(user_node, user_node_index, recv_node);
    }
  }
  MS_LOG(INFO) << "End eliminating nodes not on this process.";
}

void GraphSplitter::ReplaceOriginNodesWithRecv(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  for (const auto &op_pair_info : fused_inter_process_op_pairs) {
    const OperatorLabel &send_label = op_pair_info.first.src_label;
    const OperatorLabel &recv_label = op_pair_info.first.dst_label;
    const std::vector<FusedInterProcessOpPair> &op_pairs = op_pair_info.second;
    if (op_pairs.empty()) {
      MS_LOG(EXCEPTION) << "Fused inter-process ops should not be empty for edge " << send_label.to_string() << "->"
                        << recv_label.to_string();
    }

    const auto &fused_recv_node = std::get<1>(*op_pairs.begin());
    MS_EXCEPTION_IF_NULL(fused_recv_node);

    // Replace origin input with recv node.
    if (recv_label == this_process_label_) {
      for (const auto &send_recv_pair : op_pairs) {
        const auto &user_node = std::get<3>(send_recv_pair);
        int user_node_index = std::get<4>(send_recv_pair);

        const auto &recv_abs = fused_recv_node->abstract();
        MS_EXCEPTION_IF_NULL(recv_abs);
        // The outputs of a Recv node could be a tuple or a single tensor because it could be fused.
        if (recv_abs->isa<abstract::AbstractTuple>()) {
          int output_index = std::get<2>(send_recv_pair);
          CNodePtr tuple_get_item_node = CreateTupleGetItemNode(func_graph_, fused_recv_node, IntToSize(output_index));
          func_graph_->manager()->SetEdge(user_node, user_node_index, tuple_get_item_node);
        } else {
          func_graph_->manager()->SetEdge(user_node, user_node_index, fused_recv_node);
        }
      }
    }
  }
}

void GraphSplitter::AddSendRecvDependency(const InterProcessOpEdgesInfo &in_degree_comm_edges,
                                          const AnfNodePtrSet &send_src_nodes,
                                          const std::map<AnfNodePtr, AnfNodePtrSet> &src_nodes_to_send_nodes,
                                          const std::map<AnfNodePtr, AnfNodePtrSet> &node_dependency,
                                          std::map<AnfNodePtr, bool> *is_send_node_hung) {
  for (const auto &in_edge : in_degree_comm_edges) {
    const auto &rpc_recv_node = std::get<1>(in_edge.second);
    const auto &recv_dst_node = std::get<2>(in_edge.second);
    MS_LOG(DEBUG) << "Add dependency for RpcRecv node " << rpc_recv_node->fullname_with_scope()
                  << " with recv dst node " << recv_dst_node->fullname_with_scope();
    AnfNodePtrSet depended_nodes;
    for (const auto &send_src_node : send_src_nodes) {
      // Get minimum send src nodes set which have dependencies with RpcRecv node.
      if (node_dependency.count(recv_dst_node) != 0 && node_dependency.at(recv_dst_node).count(send_src_node) != 0) {
        depended_nodes = UpdateDependedSet(send_src_node, depended_nodes, node_dependency);
      }
    }
    MS_LOG(DEBUG) << "RpcRecv dst node " << recv_dst_node->fullname_with_scope()
                  << " depends on RpcSend src node size: " << depended_nodes.size();

    // Generate RpcSend nodes input list to add dependency with RpcRecv Nodes.
    AnfNodePtrList rpc_send_list;
    for (const auto &send_src_node : depended_nodes) {
      if (src_nodes_to_send_nodes.count(send_src_node) == 0) {
        MS_LOG(EXCEPTION) << "Send src node " << send_src_node->fullname_with_scope()
                          << " has no corresponding RpcSend nodes.";
      }
      const AnfNodePtrSet &rpc_send_nodes = src_nodes_to_send_nodes.at(send_src_node);
      for (const auto &rpc_send : rpc_send_nodes) {
        (*is_send_node_hung)[rpc_send] = false;
        rpc_send_list.emplace_back(rpc_send);
      }
    }
    if (!rpc_send_list.empty()) {
      AnfNodePtr send_node_make_tuple = CreateMakeTupleNode(func_graph_, rpc_send_list);
      MS_EXCEPTION_IF_NULL(send_node_make_tuple);
      MS_LOG(DEBUG) << "Connect " << send_node_make_tuple->fullname_with_scope() << " with RpcRecv node "
                    << rpc_recv_node->fullname_with_scope();

      auto recv_data = rpc_recv_node->cast<CNodePtr>()->inputs()[kIndex1];
      MS_EXCEPTION_IF_NULL(recv_data);

      AnfNodePtrList depend_node_inputs = {NewValueNode(prim::kPrimDepend), recv_data, send_node_make_tuple};
      auto depend_node = func_graph_->NewCNode(depend_node_inputs);
      MS_EXCEPTION_IF_NULL(depend_node);
      depend_node->set_abstract(recv_data->abstract());
      func_graph_->manager()->SetEdge(rpc_recv_node, kIndex1, depend_node);
    }
  }
}

void GraphSplitter::AddDependencyForSend(const FusedInterProcessOpPairMap &fused_inter_process_op_pairs) {
  // Connect all Send nodes to MakeTuple.
  std::vector<AnfNodePtr> fused_send_node_tuple_inputs;
  MS_EXCEPTION_IF_NULL(func_graph_);
  for (const auto &op_pair_info : fused_inter_process_op_pairs) {
    const OperatorLabel &send_label = op_pair_info.first.src_label;
    const OperatorLabel &recv_label = op_pair_info.first.dst_label;
    const std::vector<FusedInterProcessOpPair> &op_pairs = op_pair_info.second;
    if (op_pairs.empty()) {
      MS_LOG(EXCEPTION) << "Fused inter-process ops should not be empty for edge " << send_label.to_string() << "->"
                        << recv_label.to_string();
    }
    const auto &fused_send_node = std::get<0>(*op_pairs.begin());
    MS_EXCEPTION_IF_NULL(fused_send_node);
    // Make tuple all fused send nodes.
    if (send_label == this_process_label_) {
      (void)fused_send_node_tuple_inputs.emplace_back(fused_send_node);
    }
  }
  CNodePtr fused_send_make_tuple_node = CreateMakeTupleNode(func_graph_, fused_send_node_tuple_inputs);
  MS_EXCEPTION_IF_NULL(fused_send_make_tuple_node);

  // Connect fused send nodes to the output so they will not be optimized out.
  AnfNodePtr origin_output = func_graph_->output();
  if (node_labels_.count(origin_output) == 0) {
    MS_LOG(EXCEPTION) << "The origin output node " << origin_output->fullname_with_scope()
                      << " should have corresponding operator label.";
  }

  // If the output is not on this process, replace it with a fake output node.
  AnfNodePtr replaced_output = nullptr;
  if (node_labels_[origin_output] != this_process_label_) {
    replaced_output = CreateReplacedOutputNode(func_graph_, origin_output);
  } else {
    replaced_output = origin_output;
  }

  // Add dependency and replace.
  std::vector<AnfNodePtr> depend_inputs = {NewValueNode(prim::kPrimDepend), replaced_output,
                                           fused_send_make_tuple_node};
  auto final_output_node = func_graph_->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(final_output_node);
  final_output_node->set_abstract(replaced_output->abstract());
  (void)func_graph_->manager()->SetEdge(func_graph_->get_return(), 1, final_output_node);
}

bool GraphSplitter::IsNodesWithSameLabel(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  if (node_labels_.count(node1) == 0 || node_labels_.count(node2) == 0) {
    MS_LOG(EXCEPTION) << "Either 'node1': " << node1->fullname_with_scope()
                      << " or 'node2': " << node2->fullname_with_scope() << " is not marked with split label.";
  }
  return node_labels_[node1] == node_labels_[node2];
}

bool GraphSplitter::NeedSplitGraph() const {
  return std::find_if(node_labels_.begin(), node_labels_.end(), [&](const auto &node_to_label) {
           return node_to_label.second != this_process_label_;
         }) != node_labels_.end();
}

bool GraphSplitter::NodeHasLabel(const AnfNodePtr &node) { return node_labels_.count(node) != 0; }
}  // namespace parallel
}  // namespace mindspore
