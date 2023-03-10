/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/format_type/deal_ref_output.h"
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include "kernel/oplib/oplib.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
session::KernelWithIndex DealRefOutput::FindRefOriginNode(const AnfNodePtr &node) const {
  session::KernelWithIndex kernel_with_index = common::AnfAlgo::VisitKernel(node, 0);
  AnfNodePtr cur_node = kernel_with_index.first;
  size_t cur_out_index = kernel_with_index.second;
  MS_EXCEPTION_IF_NULL(cur_node);
  if (cur_node->isa<CNode>()) {
    auto cnode = cur_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    std::string op_name = common::AnfAlgo::GetCNodeName(cnode);
    auto op_info = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
    // deal ref op
    if (op_info != nullptr && op_info->is_ref()) {
      auto ref_infos = op_info->ref_infos();
      if (ref_infos.count(cur_out_index) != 0) {
        auto in_index = ref_infos.at(cur_out_index);
        if (in_index > cnode->inputs().size()) {
          MS_LOG(EXCEPTION) << "Ref op has wrong inputs: op inputs num is " << cnode->inputs().size()
                            << ", ref info is " << cur_out_index;
        }
        AnfNodePtr next_node = cnode->input(in_index + 1);
        return FindRefOriginNode(next_node);
      }
    }

    // deal special (trans,cast,reshape) op and nop-node
    if (op_name == prim::kPrimCast->name() || op_name == prim::kPrimTranspose->name() ||
        op_name == prim::kPrimTransposeD->name() || op_name == prim::kPrimReshape->name() ||
        op_name == kTransDataOpName || common::AnfAlgo::IsNopNode(cnode)) {
      AnfNodePtr next_node = cnode->input(1);
      return FindRefOriginNode(next_node);
    }
  }

  return kernel_with_index;
}

void DealRefOutput::AddRefNodePairToKernelGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                const size_t output_index, const size_t input_index) const {
  // record the ref_pair
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  session::AnfWithOutIndex final_pair = std::make_pair(cnode, output_index);
  session::KernelWithIndex kernel_with_index =
    common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cnode, input_index), 0);
  kernel_graph->AddRefCorrespondPairs(final_pair, kernel_with_index);
}

void DealRefOutput::AddRefPairToKernelGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                            const AnfNodePtr &get_item, const AnfNodePtr &final_node,
                                            size_t final_index, const session::KernelWithIndex &origin_pair) const {
  // record the ref_pair
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // if the final node is get item, means no trans or cast op is added, the final node is itself
  // so add the pair for itself, because the get item will removed later
  auto final_ref = (final_node == get_item ? cnode : final_node);
  session::AnfWithOutIndex final_pair = std::make_pair(final_ref, final_index);
  if (kernel_graph->IsInRefOutputMap(final_pair)) {
    MS_LOG(EXCEPTION) << "Ref_pair is already in ref map, node is " << final_ref->DebugString() << ", index is "
                      << final_index;
  }
  MS_LOG(DEBUG) << "Add Ref pair, final {node ptr " << final_pair.first.get() << " , info is "
                << final_pair.first->DebugString() << " , index is " << final_pair.second << "}, origin {node ptr "
                << origin_pair.first.get() << ", info is " << origin_pair.first->DebugString() << " : index "
                << origin_pair.second << "}";
  kernel_graph->AddRefCorrespondPairs(final_pair, origin_pair);
}

// if get_item is nullptr, the additional node will link to the cnode
// else the additional node will link to the get_item node (the get_item node link to cnode)
AnfNodePtr DealRefOutput::AddAdditionalToRefOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                   size_t output_index, size_t input_index,
                                                   const AnfNodePtr &get_item) const {
  AnfNodePtr final_node = (get_item == nullptr ? cnode : get_item);
  bool need_refresh_ref_addr = false;
  size_t final_index = output_index;
  AnfNodePtr input_node = common::AnfAlgo::GetInputNode(cnode, input_index);
  session::KernelWithIndex origin_pair = FindRefOriginNode(input_node);
  MS_EXCEPTION_IF_NULL(origin_pair.first);
  MS_LOG(DEBUG) << "DealRefTransAndCast the node input index " << input_index << ", find origin op is "
                << origin_pair.first->DebugString() << ", index is " << origin_pair.second;
  auto origin_format = AnfAlgo::GetOutputFormat(origin_pair.first, origin_pair.second);
  auto origin_type = AnfAlgo::GetOutputDeviceDataType(origin_pair.first, origin_pair.second);
  auto cur_format = AnfAlgo::GetOutputFormat(cnode, output_index);
  auto cur_type = AnfAlgo::GetOutputDeviceDataType(cnode, output_index);
  auto cur_shape = common::AnfAlgo::GetOutputInferShape(cnode, output_index);
  auto detail_shape = AnfAlgo::GetOutputDetailShape(cnode, output_index);
  // insert trans
  if (origin_format != cur_format && cur_shape.size() > 1) {
    auto kernel_select = std::make_shared<KernelSelect>();
    if (cur_format != kOpFormat_DEFAULT) {
      auto cur_reshape_type = AnfAlgo::GetOutputReshapeType(cnode, output_index);
      final_node = AddTransOpNodeToGraphWithFormat(func_graph, final_node, final_node, kernel_select, cur_format,
                                                   kOpFormat_DEFAULT, cur_reshape_type, cur_type);
    }
    if (origin_format != kOpFormat_DEFAULT) {
      int64_t groups = common::AnfAlgo::GetAttrGroups(origin_pair.first, origin_pair.second);
      auto origin_reshape_type = AnfAlgo::GetOutputReshapeType(origin_pair.first, origin_pair.second);
      final_node = AddTransOpNodeToGraphWithFormat(func_graph, final_node, final_node, kernel_select, kOpFormat_DEFAULT,
                                                   origin_format, origin_reshape_type, cur_type, groups);
    }
    final_index = 0;
    need_refresh_ref_addr = true;
    MS_EXCEPTION_IF_NULL(final_node);
    MS_LOG(INFO) << "DealRefTransAndCast add trans op, op debug info is " << final_node->DebugString();
  }
  // insert cast
  if (origin_type != cur_type) {
    final_node =
      AddCastOpNodeToGraph(func_graph, final_node, cnode, origin_format, cur_type, origin_type, detail_shape, cur_type);
    MS_EXCEPTION_IF_NULL(final_node);
    final_node->set_scope(cnode->scope());
    final_index = 0;
    need_refresh_ref_addr = true;
    MS_LOG(INFO) << "DealRefTransAndCast add cast op, op debug info is " << final_node->DebugString();
  }
  // add ref pair
  auto ref_final_node =
    common::AnfAlgo::GetCNodeName(final_node) == kReshapeOpName ? final_node->cast<CNodePtr>()->input(1) : final_node;
  AddRefPairToKernelGraph(func_graph, cnode, get_item, ref_final_node, final_index, origin_pair);
  if (need_refresh_ref_addr) {
    AddRefNodePairToKernelGraph(func_graph, cnode, output_index, input_index);
  }
  // insert depend
  if (origin_format != cur_format || origin_type != cur_type) {
    final_node = MakeDependency(get_item, final_node, cnode, func_graph);
    MS_LOG(INFO) << "DealRefTranshwAndCast add denpend, op debug info is " << final_node->DebugString();
  }
  return final_node;
}

CNodePtr DealRefOutput::MakeDependency(const AnfNodePtr &get_item, const AnfNodePtr &final_node, const CNodePtr &cnode,
                                       const FuncGraphPtr &func_graph) const {
  std::vector<AnfNodePtr> depend_nodes;
  if (get_item != nullptr) {
    depend_nodes = std::vector<AnfNodePtr>{NewValueNode(prim::kPrimDepend), get_item, final_node};
  } else {
    depend_nodes = std::vector<AnfNodePtr>{NewValueNode(prim::kPrimDepend), cnode, final_node};
  }
  return func_graph->NewCNode(depend_nodes);
}

AnfNodePtr DealRefOutput::DealRefForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &orig_cnode,
                                                   const std::shared_ptr<kernel::OpInfo> &op_info) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto cnode = orig_cnode;
  auto update_states = common::AnfAlgo::GetUpdateStateUsers(manager, orig_cnode);
  if (!update_states.empty()) {
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    cnode = NewCNode(orig_cnode, kernel_graph);
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_inputs(orig_cnode->inputs());
    for (auto &update_state : update_states) {
      manager->SetEdge(update_state.first, update_state.second, cnode);
    }
  }
  MS_EXCEPTION_IF_NULL(op_info);
  auto ref_infos = op_info->ref_infos();
  std::vector<AnfNodePtr> make_tuple_inputs;
  AbstractBasePtrList abstract_list;
  (void)make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    AnfNodePtr final_node = CreatTupleGetItemNode(func_graph, cnode, output_index);
    // deal with ref output
    if (ref_infos.count(output_index) != 0) {
      auto input_index = ref_infos.at(output_index);
      final_node = AddAdditionalToRefOutput(func_graph, cnode, output_index, input_index, final_node);
    }
    MS_EXCEPTION_IF_NULL(final_node);
    abstract_list.push_back(final_node->abstract());
    make_tuple_inputs.push_back(final_node);
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return make_tuple;
}

AnfNodePtr DealRefOutput::DealRefSingleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                              const std::shared_ptr<kernel::OpInfo> &op_info) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  auto ref_infos = op_info->ref_infos();
  if (ref_infos.empty()) {
    return nullptr;
  }
  auto ref_info = *(ref_infos.begin());
  if (ref_info.second > cnode->inputs().size()) {
    MS_LOG(EXCEPTION) << "Ref op has wrong inputs: op inputs num is " << cnode->inputs().size() << ", ref info is "
                      << ref_info.second;
  }
  return AddAdditionalToRefOutput(func_graph, cnode, ref_info.first, ref_info.second, nullptr);
}

const BaseRef DealRefOutput::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(UnVisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

void DealRefOutput::DealBroadCastAsRef(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const {
  if (common::AnfAlgo::GetCNodeName(cnode) == kBroadcastOpName) {
    auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t i = 0; i < input_size; ++i) {
      auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i, true);
      auto input_node = input_node_with_index.first;
      MS_EXCEPTION_IF_NULL(input_node);
      MS_LOG(INFO) << "origin node:" << input_node->fullname_with_scope();
      AddRefPairToKernelGraph(func_graph, cnode, nullptr, cnode, i, input_node_with_index);
    }
  }
}

const AnfNodePtr DealRefOutput::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfUtils::IsRealCNodeKernel(cnode)) {
    return nullptr;
  }

  DealBroadCastAsRef(graph, cnode);

  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto op_info = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
  if (op_info == nullptr || !op_info->is_ref()) {
    return nullptr;
  }
  if (op_info->is_ref()) {
    auto type = cnode->Type();
    MS_EXCEPTION_IF_NULL(type);
    if (!type->isa<Tuple>()) {
      return DealRefSingleOutput(graph, cnode, op_info);
    } else {
      return DealRefForMultipleOutput(graph, cnode, op_info);
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
