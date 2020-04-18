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

#include "pre_activate/ascend/format_type/deal_ref_trans_and_cast.h"
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include "kernel/oplib/oplib.h"
#include "session/anf_runtime_algorithm.h"
#include "session/kernel_graph.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
session::KernelWithIndex FindRefOriginNode(const AnfNodePtr &node) {
  session::KernelWithIndex kernel_with_index = AnfAlgo::VisitKernel(node, 0);
  AnfNodePtr cur_node = kernel_with_index.first;
  size_t cur_out_index = kernel_with_index.second;
  if (cur_node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    std::string op_name = AnfAlgo::GetCNodeName(cnode);
    auto op_info = mindspore::kernel::OpLib::FindOp(op_name, kernel::kTBE);
    // deal ref op
    if (op_info->is_ref()) {
      auto ref_infos = op_info->ref_infos();
      if (ref_infos.count(cur_out_index) != 0) {
        auto in_index = ref_infos.at(cur_out_index);
        if (in_index > cnode->inputs().size()) {
          MS_LOG(EXCEPTION) << "ref op has wrong inputs: op inputs num is " << cnode->inputs().size()
                            << ", ref info is " << cur_out_index;
        }
        AnfNodePtr next_node = cnode->input(in_index + 1);
        return FindRefOriginNode(next_node);
      }
    }

    // deal special (trans,cast,reshape) op
    if (op_name == prim::kPrimCast->name() || op_name == prim::kPrimTranspose->name() ||
        op_name == prim::kPrimReshape->name() || op_name == kTransDataOpName) {
      AnfNodePtr next_node = cnode->input(1);
      return FindRefOriginNode(next_node);
    }
  }

  return kernel_with_index;
}

void AddRefPairToKernelGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &get_item,
                             const AnfNodePtr &final_node, size_t final_index,
                             const session::KernelWithIndex &origin_pair) {
  // record the ref_pair
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // if the final node is get item, means no trans or cast op is added, the final node is itself
  // so add the pair for itself, because the get item will removed later
  auto final_ref = (final_node == get_item ? cnode : final_node);
  session::AnfWithOutIndex final_pair = std::make_pair(final_ref, final_index);
  if (kernel_graph->IsInRefOutputMap(final_pair)) {
    MS_LOG(EXCEPTION) << "ref_pair is already in ref map, node is " << final_ref->DebugString() << ", index is "
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
AnfNodePtr AddAdditionalToRefOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t output_index,
                                    size_t input_index, const AnfNodePtr &get_item) {
  AnfNodePtr final_node = (get_item == nullptr ? cnode : get_item);
  size_t final_index = output_index;
  AnfNodePtr input_node = cnode->input(input_index + 1);
  session::KernelWithIndex origin_pair;
  origin_pair = FindRefOriginNode(input_node);
  MS_EXCEPTION_IF_NULL(origin_pair.first);
  if (!origin_pair.first->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "ref op origin node is not parameter";
  }
  MS_LOG(DEBUG) << "DealRefTransAndCast the node input index " << input_index << ", find origin op is "
                << origin_pair.first->DebugString() << ", index is " << origin_pair.second;
  auto origin_format = AnfAlgo::GetOutputFormat(origin_pair.first, origin_pair.second);
  auto origin_type = AnfAlgo::GetOutputDeviceDataType(origin_pair.first, origin_pair.second);
  auto cur_format = AnfAlgo::GetOutputFormat(cnode, output_index);
  auto cur_type = AnfAlgo::GetOutputDeviceDataType(cnode, output_index);
  auto cur_shape = AnfAlgo::GetOutputInferShape(cnode, output_index);
  // insert trans
  if (origin_format != cur_format && cur_shape.size() > 1) {
    auto kernel_select = std::make_shared<KernelSelect>();
    final_node = AddTransOpNodeToGraph(func_graph, final_node, kernel_select, 0, cur_format, origin_format,
                                       kTransDataOpName, false);
    final_index = 0;
    MS_EXCEPTION_IF_NULL(final_node);
    MS_LOG(INFO) << "DealRefTransAndCast add trans op, op debug info is " << final_node->DebugString();
  }
  // insert cast
  if (origin_type != cur_type) {
    final_node =
      AddCastOpNodeToGraph(func_graph, final_node, origin_format, cur_type, origin_type, cur_shape, cur_type);
    MS_EXCEPTION_IF_NULL(final_node);
    final_node->set_scope(cnode->scope());
    final_index = 0;
    MS_LOG(INFO) << "DealRefTransAndCast add cast op, op debug info is " << final_node->DebugString();
  }
  // add ref pair
  AddRefPairToKernelGraph(func_graph, cnode, get_item, final_node, final_index, origin_pair);
  // insert depend
  if (origin_format != cur_format || origin_type != cur_type) {
    std::vector<AnfNodePtr> depend_nodes{NewValueNode(prim::kPrimDepend), cnode, final_node};
    final_node = func_graph->NewCNode(depend_nodes);
    MS_LOG(INFO) << "DealRefTransAndCast add denpend, op debug info is " << final_node->DebugString();
  }

  return final_node;
}
AnfNodePtr DealRefForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                    const std::shared_ptr<kernel::OpInfo> &op_info) {
  auto ref_infos = op_info->ref_infos();
  std::vector<AnfNodePtr> make_tuple_inputs;
  AbstractBasePtrList abstract_list;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(cnode); ++output_index) {
    AnfNodePtr final_node = CreatTupleGetItemNode(func_graph, cnode, output_index);
    // deal with ref output
    if (ref_infos.count(output_index) != 0) {
      auto input_index = ref_infos.at(output_index);
      final_node = AddAdditionalToRefOutput(func_graph, cnode, output_index, input_index, final_node);
    }
    abstract_list.push_back(final_node->abstract());
    make_tuple_inputs.push_back(final_node);
  }
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return make_tuple;
}

AnfNodePtr DealRefSigleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                              const std::shared_ptr<kernel::OpInfo> &op_info) {
  auto ref_infos = op_info->ref_infos();
  for (const auto &ref_info : ref_infos) {
    if (ref_info.second > cnode->inputs().size()) {
      MS_LOG(EXCEPTION) << "ref op has wrong inputs: op inputs num is " << cnode->inputs().size() << ", ref info is "
                        << ref_info.second;
    }
    return AddAdditionalToRefOutput(func_graph, cnode, ref_info.first, ref_info.second, nullptr);
  }
  return nullptr;
}
}  // namespace

const BaseRef DealRefTransAndCast::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(UnVisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr DealRefTransAndCast::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::IsRealCNodeKernel(cnode)) {
    return nullptr;
  }
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  auto op_info = mindspore::kernel::OpLib::FindOp(op_name, kernel::kTBE);
  if (op_info == nullptr || !op_info->is_ref()) {
    return nullptr;
  }
  if (op_info->is_ref()) {
    if (!cnode->Type()->isa<Tuple>()) {
      return DealRefSigleOutput(graph, cnode, op_info);
    } else {
      return DealRefForMultipleOutput(graph, cnode, op_info);
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
