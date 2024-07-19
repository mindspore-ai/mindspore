/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "ops/sequence_ops.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "ops/op_def.h"
#include "plugin/device/ascend/optimizer/format_type/utils.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "kernel/oplib/oplib.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/helper.h"
#include "transform/acl_ir/ge_adapter_info.h"

namespace mindspore {
namespace opt {
namespace {
std::unordered_map<size_t, size_t> GetRefInfoMaps(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::unordered_map<size_t, size_t> ref_infos;
  auto kernel_type = AnfAlgo::GetKernelType(cnode);
  if (kernel_type == KernelType::UNKNOWN_KERNEL_TYPE) {
    return ref_infos;
  }

  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if (kernel_type == KernelType::ACL_KERNEL) {
    auto info = transform::GeAdapterManager::GetInstance().GetInfo(op_name, true);
    if (info == nullptr) {
      return ref_infos;
    }

    ref_infos = info->GetRefMappingInfo();
  } else if (kernel_type == KernelType::OPAPI_KERNEL) {
    mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
    if (op_def == nullptr) {
      return ref_infos;
    }
    for (size_t i = 0; i < op_def->returns_.size(); ++i) {
      if (op_def->returns_[i].inplace_input_index_ != -1) {
        ref_infos[i] = op_def->returns_[i].inplace_input_index_;
      }
    }
  }

  return ref_infos;
}

AnfNodePtr GetRefInputNode(const CNodePtr &cnode, const size_t cur_out_index) {
  MS_EXCEPTION_IF_NULL(cnode);

  auto ref_infos = GetRefInfoMaps(cnode);
  if (!ref_infos.empty()) {
    if (ref_infos.count(cur_out_index) != 0) {
      auto in_index = ref_infos.at(cur_out_index);
      if (in_index > cnode->size()) {
        MS_LOG(EXCEPTION) << "Ref op has wrong inputs: op inputs num is " << cnode->size() << ", ref info is "
                          << cur_out_index;
      }
      return cnode->input(in_index + 1);
    }
  }

  return nullptr;
}

session::KernelWithIndex FindRefOriginNode(const AnfNodePtr &node) {
  session::KernelWithIndex kernel_with_index = common::AnfAlgo::VisitKernel(node, 0);
  AnfNodePtr cur_node = kernel_with_index.first;
  size_t cur_out_index = kernel_with_index.second;
  MS_EXCEPTION_IF_NULL(cur_node);
  if (cur_node->isa<CNode>()) {
    auto cnode = cur_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    std::string op_name = common::AnfAlgo::GetCNodeName(cnode);
    // deal special (identity,cast,reshape) op and nop-node
    if (op_name == prim::kPrimCast->name() || op_name == prim::kPrimIdentity->name() ||
        op_name == prim::kPrimReshape->name() || common::AnfAlgo::IsNopNode(cnode)) {
      AnfNodePtr next_node = cnode->input(kIndex1);
      return FindRefOriginNode(next_node);
    }

    AnfNodePtr next_node = GetRefInputNode(cnode, cur_out_index);
    if (next_node) {
      return FindRefOriginNode(next_node);
    }
  }

  return kernel_with_index;
}
}  // namespace

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
    MS_LOG(INTERNAL_EXCEPTION) << "Ref_pair is already in ref map, node is " << final_ref->DebugString()
                               << ", index is " << final_index;
  }
  MS_LOG(DEBUG) << "Add Ref pair, final {node ptr " << final_pair.first.get() << " , info is "
                << final_pair.first->DebugString() << " , index is " << final_pair.second << "}, origin {node ptr "
                << origin_pair.first.get() << ", info is " << origin_pair.first->DebugString() << " : index "
                << origin_pair.second << "}";
  kernel_graph->AddRefCorrespondPairs(final_pair, origin_pair);
}

// if get_item is nullptr, the additional node(identity) will link to the cnode
// else the additional node(identity) will link to the get_item node (the get_item node link to cnode)
AnfNodePtr DealRefOutput::AddAdditionalToRefOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                   size_t output_index, size_t input_index,
                                                   const AnfNodePtr &get_item) const {
  AnfNodePtr final_node = (get_item == nullptr ? cnode : get_item);
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
  bool need_refresh_ref_addr = (origin_format != cur_format && cur_shape.size() > 1) || origin_type != cur_type;
  // insert identity
  if (need_refresh_ref_addr) {
    auto identity_node = NewCNode({NewValueNode(std::make_shared<Primitive>(kIdentityOpName)), final_node}, func_graph);
    identity_node->set_scope(cnode->scope());
    abstract::AbstractTensorPtr abs =
      std::make_shared<abstract::AbstractTensor>(TypeIdToType(origin_type), detail_shape);
    identity_node->set_abstract(abs);
    // set kernel build info
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetKernelType(KernelType::ACL_KERNEL);
    builder.SetInputsFormat({cur_format});
    builder.SetOutputsFormat({origin_format});
    builder.SetInputsReshapeType({});
    builder.SetOutputsReshapeType({});
    builder.SetInputsDeviceType({cur_type});
    builder.SetOutputsDeviceType({origin_type});
    builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), identity_node.get());

    final_node = identity_node;
    MS_LOG(INFO) << "DealRefOutput add Identity node " << final_node->fullname_with_scope();
  }

  // add ref pair
  AddRefPairToKernelGraph(func_graph, cnode, get_item, final_node, output_index, origin_pair);
  if (need_refresh_ref_addr) {
    AddRefNodePairToKernelGraph(func_graph, cnode, output_index, input_index);
    // insert depend
    final_node = MakeDependency(get_item, final_node, cnode, func_graph);
    MS_LOG(INFO) << "DealRefOutput add Denpend node " << final_node->fullname_with_scope();
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
                                                   const std::unordered_map<size_t, size_t> &ref_infos) const {
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
                                              const std::unordered_map<size_t, size_t> &ref_infos) const {
  MS_EXCEPTION_IF_NULL(cnode);
  auto ref_info = *(ref_infos.begin());
  if (ref_info.second > cnode->size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Ref op has wrong inputs: op inputs num is " << cnode->size() << ", ref info is "
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

  if (AnfAlgo::IsKernelSelectBackoffOp(cnode)) {
    return nullptr;
  }

  DealBroadCastAsRef(graph, cnode);

  auto ref_infos = GetRefInfoMaps(cnode);
  if (ref_infos.empty()) {
    return nullptr;
  }

  auto type = cnode->Type();
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<Tuple>()) {
    return DealRefSingleOutput(graph, cnode, ref_infos);
  } else {
    return DealRefForMultipleOutput(graph, cnode, ref_infos);
  }

  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
