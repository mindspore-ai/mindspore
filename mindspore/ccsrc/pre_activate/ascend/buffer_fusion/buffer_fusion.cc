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
#include "pre_activate/ascend/buffer_fusion/buffer_fusion.h"

#include <vector>
#include <tuple>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>

#include "kernel/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"
#include "device/kernel_info.h"
#include "utils/context/ms_context.h"
#include "pre_activate/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
namespace {
const int8_t MAX_PATTERN_SIZE = 7;
const int8_t MIN_PATTERN_SIZE = 2;
const int8_t ELTWISE_INPUT_SIZE = 2;
const int8_t ELTWISE_USE = 1;
const int8_t MULTI_ELTWISE_USE = 2;
const int8_t MAX_MULTI_ELTWISE_SIZE = 4;
const int8_t MAX_PURE_BUFFER_SUCC_SIZE = 3;
constexpr auto kOpAttrFusionId = "fusion_id";

#ifdef DEBUG
std::string GetFusionTypeName(const kernel::FusionType &type) {
  switch (type) {
    case kernel::FusionType::COMMREDUCE:
      return "COMMREDUCE";
    case kernel::FusionType::SEGMENT:
      return "SEGMENT";
    case kernel::FusionType::ELEMWISE:
      return "ELEMWISE";
    case kernel::FusionType::CONVLUTION:
      return "CONVLUTION";
    case kernel::FusionType::OPAQUE:
      return "OPAQUE";
    default:
      return "OPAQUE";
  }
}

void DumpFusionScopeInfo(const kernel::FusionScopeInfo &info) {
  MS_LOG(INFO) << "=== Dump FusionScopeInfo start id: " << info.scope_id;
  for (auto &node : info.input_nodes) {
    MS_LOG(INFO) << "=== Input: " << node->DebugString();
  }
  for (auto &node : info.output_nodes) {
    MS_LOG(INFO) << "=== Output: " << node->DebugString();
  }
  for (auto &node : info.compute_nodes) {
    MS_LOG(INFO) << "=== Compute: (" << node->DebugString() << ")-(" << GetFusionTypeName(AnfAlgo::GetFusionType(node))
                 << ")";
  }
  MS_LOG(INFO) << "=== Dump FusionScopeInfo end";
}
#endif

bool CheckEltWiseNode(FuncGraphManager *manager, std::unordered_set<AnfNodePtr> *record, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(record);
  auto user_nodes = manager->node_users()[node];
  return (AnfAlgo::GetKernelType(node) == KernelType::TBE_KERNEL &&
          AnfAlgo::GetFusionType(node) == kernel::FusionType::ELEMWISE &&
          (user_nodes.size() <= ELTWISE_USE || record->size() == 0));
}

// Common method to check for predecessors and successors in a fusion pattern
std::tuple<bool, CNodePtr> FindPredAndSuccEltWiseNodes(const int8_t &max_size, FuncGraphManager *manager,
                                                       std::unordered_set<AnfNodePtr> *visited_set,
                                                       std::deque<AnfNodePtr> *todo,
                                                       std::unordered_set<AnfNodePtr> *record, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(visited_set);
  MS_EXCEPTION_IF_NULL(todo);
  MS_EXCEPTION_IF_NULL(record);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr new_node = node;
  if (new_node->inputs().size() < ELTWISE_INPUT_SIZE) {
    return std::make_tuple(false, new_node);
  }
  int8_t index = 1;
  auto &users = manager->node_users();
  while (CheckEltWiseNode(manager, record, new_node)) {
    (void)record->insert(new_node);
    (void)visited_set->insert(new_node);
    (void)todo->insert(todo->end(), new_node->inputs().begin() + 1, new_node->inputs().end());

    auto cnode = new_node->input(1);
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->isa<CNode>()) {
      return std::make_tuple(false, new_node);
    }
    new_node = cnode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(new_node);

    if (!AnfAlgo::IsRealKernel(new_node) || new_node->inputs().size() < ELTWISE_INPUT_SIZE ||
        users[(new_node)].size() >= MULTI_ELTWISE_USE || visited_set->find(new_node) != visited_set->end()) {
      return std::make_tuple(false, new_node);
    }

    if (index >= max_size) {
      break;
    }
    index++;
  }
  return std::make_tuple(true, new_node);
}

std::tuple<bool, CNodePtr> MatchGeneralPattern(FuncGraphManager *manager, std::unordered_set<AnfNodePtr> *record,
                                               std::unordered_set<AnfNodePtr> *visited_set,
                                               std::deque<AnfNodePtr> *todo, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(record);
  MS_EXCEPTION_IF_NULL(visited_set);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(todo);
  CNodePtr new_node = node;
  auto &users = manager->node_users();
  if (users[(new_node)].size() >= MULTI_ELTWISE_USE) {
    return std::make_tuple(false, new_node);
  }

  (void)record->insert(node);
  (void)visited_set->insert(node);
  (void)todo->insert(todo->end(), new_node->inputs().begin() + 1, new_node->inputs().end());

  if (node->inputs().size() < 2) {
    return std::make_tuple(false, new_node);
  }
  // only check the first real input, will check all
  auto cnode = node->input(1);
  MS_EXCEPTION_IF_NULL(cnode);
  if (!cnode->isa<CNode>()) {
    return std::make_tuple(false, new_node);
  }
  new_node = cnode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(new_node);

  if (!AnfAlgo::IsRealKernel(new_node) || users[(new_node)].size() >= MULTI_ELTWISE_USE ||
      visited_set->find(new_node) != visited_set->end()) {
    return std::make_tuple(false, new_node);
  }
  return std::make_tuple(true, new_node);
}

CNodePtr FindFusionAnfNode(FuncGraphManager *manager, std::unordered_set<AnfNodePtr> *visited_set,
                           std::unordered_set<AnfNodePtr> *record, std::deque<AnfNodePtr> *todo, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(visited_set);
  MS_EXCEPTION_IF_NULL(record);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(todo);
  // find fusion pattern predecessor nodes
  auto ret = FindPredAndSuccEltWiseNodes(MAX_MULTI_ELTWISE_SIZE, manager, visited_set, todo, record, node);
  auto new_node = std::get<1>(ret);
  auto node_use_size = manager->node_users()[new_node].size();
  if (!std::get<0>(ret) || (record->size() > 1 && node_use_size > 1) || record->size() >= MAX_MULTI_ELTWISE_SIZE ||
      AnfAlgo::GetKernelType(new_node) != KernelType::TBE_KERNEL) {
    return new_node;
  }

  // key of fusion precessor
  auto node_fusion_type = AnfAlgo::GetFusionType(new_node);
  switch (node_fusion_type) {
    case kernel::FusionType::COMMREDUCE:
    case kernel::FusionType::SEGMENT:
      ret = MatchGeneralPattern(manager, record, visited_set, todo, new_node);
      new_node = std::get<1>(ret);
      if (!std::get<0>(ret)) {
        return new_node;
      }
      break;
    case kernel::FusionType::ELEMWISE:
      return new_node;
    // -fallthrough to default and return
    case kernel::FusionType::CONVLUTION:
      (void)record->insert(new_node);
    default:
      (void)visited_set->insert(new_node);
      if (new_node != nullptr) {
        (void)todo->insert(todo->end(), new_node->inputs().begin() + 1, new_node->inputs().end());
      }
      return new_node;
  }
  // find fusion pattern successor nodes
  ret = FindPredAndSuccEltWiseNodes(MAX_PURE_BUFFER_SUCC_SIZE, manager, visited_set, todo, record, new_node);
  return std::get<1>(ret);
}

CNodePtr CreateFusionOp(const std::vector<AnfNodePtr> &inputs_list, const std::vector<AnfNodePtr> &outputs_list,
                        const std::vector<AnfNodePtr> &anf_nodes, session::KernelGraph *kernel_graph) {
  MS_LOG(DEBUG) << "Start Create FusionOp Kernel";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::string fusion_op_name = "FusionOp";
  for (auto node : anf_nodes) {
    fusion_op_name += '_' + AnfAlgo::GetCNodeName(node);
  }
  auto fusion_op = std::make_shared<Primitive>(fusion_op_name);
  MS_EXCEPTION_IF_NULL(fusion_op);

  std::vector<std::string> input_names;
  for (uint8_t i = 0; i < inputs_list.size(); i++) {
    input_names.emplace_back("input" + std::to_string(i));
  }
  std::vector<std::string> output_names;
  for (uint8_t i = 0; i < outputs_list.size(); i++) {
    output_names.emplace_back("output" + std::to_string(i));
  }

  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  fusion_op->set_attr("input_names", input_names_v);
  fusion_op->set_attr("output_names", output_names_v);
  std::vector<AnfNodePtr> fusion_inputs_list = inputs_list;
  auto value_node = std::make_shared<ValueNode>(fusion_op);
  (void)fusion_inputs_list.insert(fusion_inputs_list.begin(), value_node);
  auto buffer_fusion_kernel = kernel_graph->NewCNode(fusion_inputs_list);
  if (buffer_fusion_kernel == nullptr) {
    MS_LOG(EXCEPTION) << "New FusionOp kernel failed!";
  }
  buffer_fusion_kernel->set_scope((anf_nodes.back())->scope());

  return buffer_fusion_kernel;
}

kernel::KernelBuildInfoPtr CreateFusionOpKernelInfo(const std::vector<AnfNodePtr> &inputs_list,
                                                    const std::vector<AnfNodePtr> &outputs_list) {
  MS_LOG(DEBUG) << "Start Create Kernel Info";
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  // inputs format and data type
  std::vector<std::string> inputs_format;
  std::vector<TypeId> inputs_data_type;
  for (const auto &input : inputs_list) {
    auto real_input = AnfAlgo::VisitKernel(input, 0);
    inputs_format.push_back(AnfAlgo::GetOutputFormat(real_input.first, real_input.second));
    inputs_data_type.push_back(AnfAlgo::GetOutputDeviceDataType(real_input.first, real_input.second));
  }
  // outputs format and data type
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_data_type;
  for (const auto &output : outputs_list) {
    if (AnfAlgo::GetCNodeName(output) == prim::kPrimTupleGetItem->name()) {
      auto tuple_getitem = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_getitem);
      outputs_format.push_back(AnfAlgo::GetOutputFormat(
        tuple_getitem->input(1), IntToSize(GetValue<int>(GetValueNode(tuple_getitem->input(2))))));
      outputs_data_type.push_back(AnfAlgo::GetOutputDeviceDataType(
        tuple_getitem->input(1), IntToSize(GetValue<int>(GetValueNode(tuple_getitem->input(2))))));
    } else {
      outputs_format.push_back(AnfAlgo::GetOutputFormat(output, 0));
      outputs_data_type.push_back(AnfAlgo::GetOutputDeviceDataType(output, 0));
    }
  }
  builder.SetInputsFormat(inputs_format);
  builder.SetInputsDeviceType(inputs_data_type);
  builder.SetOutputsFormat(outputs_format);
  builder.SetOutputsDeviceType(outputs_data_type);
  builder.SetKernelType(KernelType::TBE_KERNEL);
  return builder.Build();
}

AnfNodePtr CreateTupleGetItem(const AnfNodePtr &buffer_fusion_kernel, session::KernelGraph *kernel_graph,
                              size_t output_index) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<AnfNodePtr> tuple_getitem_inputs_list;
  auto value = std::make_shared<ValueNode>(prim::kPrimTupleGetItem);
  MS_EXCEPTION_IF_NULL(value);
  auto idx = NewValueNode(SizeToInt(output_index));
  MS_EXCEPTION_IF_NULL(idx);
  int temp = SizeToInt(output_index);
  auto imm = std::make_shared<Int32Imm>(temp);
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  tuple_getitem_inputs_list.push_back(value);
  tuple_getitem_inputs_list.push_back(buffer_fusion_kernel);
  tuple_getitem_inputs_list.push_back(idx);
  auto tuple_item = kernel_graph->NewCNode(tuple_getitem_inputs_list);
  MS_EXCEPTION_IF_NULL(tuple_item);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(buffer_fusion_kernel, output_index)},
                                      {AnfAlgo::GetOutputInferShape(buffer_fusion_kernel, output_index)},
                                      tuple_item.get());
  return tuple_item;
}

void ReplaceInputNodeInOtherFusionScope(std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos,
                                        int32_t fusion_id, const AnfNodePtr &output_item,
                                        const AnfNodePtr &replace_item) {
  for (int32_t id = fusion_id + 1; id <= SizeToInt(buffer_fusion_infos->size()); ++id) {
    auto itr = std::find((*buffer_fusion_infos)[id].inputs_list.begin(), (*buffer_fusion_infos)[id].inputs_list.end(),
                         output_item);
    if (itr != (*buffer_fusion_infos)[id].inputs_list.end()) {
      MS_LOG(DEBUG) << "replace input of other pattern, id = " << id;
      *itr = replace_item;
    }
  }
}

void ReplaceOldNode(std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos, int32_t fusion_id,
                    const AnfNodePtr &buffer_fusion_kernel, session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto buffer_fusion_info = (*buffer_fusion_infos)[fusion_id];
  if (buffer_fusion_info.outputs_list.size() == 1) {  // single output
    (void)manager->Replace(buffer_fusion_info.outputs_list[0], buffer_fusion_kernel);
    ReplaceInputNodeInOtherFusionScope(buffer_fusion_infos, fusion_id, buffer_fusion_info.outputs_list[0],
                                       buffer_fusion_kernel);
  } else {  // multiple output
    for (size_t index = 0; index < buffer_fusion_info.outputs_list.size(); ++index) {
      auto tuple_item = CreateTupleGetItem(buffer_fusion_kernel, kernel_graph, index);
      (void)manager->Replace(buffer_fusion_info.outputs_list[index], tuple_item);
      ReplaceInputNodeInOtherFusionScope(buffer_fusion_infos, fusion_id, buffer_fusion_info.outputs_list[index],
                                         tuple_item);
    }
  }
}

void GetFusionScopeComputeNodeList(session::KernelGraph *kernel_graph,
                                   std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto nodes = TopoSort(kernel_graph->get_return());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::IsRealCNodeKernel(node) && AnfAlgo::HasNodeAttr(kOpAttrFusionId, node)) {
      auto fusion_id = AnfAlgo::GetNodeAttr<int32_t>(node, kOpAttrFusionId);
      (*buffer_fusion_infos)[fusion_id].anf_nodes.push_back(node);
    }
  }
}

void GetFusionScopeInputNodeList(const session::KernelGraph &kernel_graph,
                                 std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    auto fusion_id = buffer_fusion_info.first;
    auto fusion_info = buffer_fusion_info.second;
    for (const auto &node : fusion_info.anf_nodes) {
      auto cnode = node->cast<CNodePtr>();
      for (size_t idx = 1; idx < cnode->inputs().size(); ++idx) {
        auto real_input = AnfAlgo::VisitKernel(cnode->input(idx), 0);
        if (std::find(fusion_info.anf_nodes.begin(), fusion_info.anf_nodes.end(), real_input.first) ==
            fusion_info.anf_nodes.end()) {
          if (std::find((*buffer_fusion_infos)[fusion_id].inputs_list.begin(),
                        (*buffer_fusion_infos)[fusion_id].inputs_list.end(),
                        cnode->input(idx)) == (*buffer_fusion_infos)[fusion_id].inputs_list.end()) {
            (*buffer_fusion_infos)[fusion_id].inputs_list.push_back(cnode->input(idx));
          }
        }
      }
    }
  }
}

bool TupleGetitemNodeCompare(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  auto getitem1 = node1->cast<CNodePtr>();
  auto getitem2 = node2->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(getitem1);
  MS_EXCEPTION_IF_NULL(getitem2);
  auto output_idx1 = GetValue<int>(GetValueNode(getitem1->input(2)));
  auto output_idx2 = GetValue<int>(GetValueNode(getitem2->input(2)));
  return output_idx1 < output_idx2;
}

void GetFusionScopeOutputNodeList(session::KernelGraph *kernel_graph,
                                  std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    auto fusion_id = buffer_fusion_info.first;
    auto fusion_info = buffer_fusion_info.second;
    for (const auto &node : fusion_info.anf_nodes) {
      if (AnfAlgo::GetOutputTensorNum(node) == 1) {
        for (auto use_node : manager->node_users()[node]) {
          if (std::find(fusion_info.anf_nodes.begin(), fusion_info.anf_nodes.end(), use_node.first) ==
              fusion_info.anf_nodes.end()) {
            (*buffer_fusion_infos)[fusion_id].outputs_list.push_back(node);
            break;
          }
        }
      } else {
        int prev_idx = 0;
        std::vector<AnfNodePtr> tuple_getitem_nodes;
        std::transform(manager->node_users()[node].begin(), manager->node_users()[node].end(),
                       std::back_inserter(tuple_getitem_nodes),
                       [](const std::pair<AnfNodePtr, int> &use_node) { return use_node.first; });
        std::sort(tuple_getitem_nodes.begin(), tuple_getitem_nodes.end(), TupleGetitemNodeCompare);
        for (auto getitem : tuple_getitem_nodes) {
          auto getitem_ptr = getitem->cast<CNodePtr>();
          auto input2 = getitem_ptr->input(2);
          auto output_idx = GetValue<int>(GetValueNode(input2));
          for (int stub_idx = prev_idx; stub_idx < output_idx; ++stub_idx) {
            auto stub_node = CreateTupleGetItem(node, kernel_graph, IntToSize(stub_idx));
            (*buffer_fusion_infos)[fusion_id].outputs_list.push_back(stub_node);
          }
          prev_idx = output_idx + 1;
          for (auto item_use_node : manager->node_users()[getitem]) {
            if (std::find(fusion_info.anf_nodes.begin(), fusion_info.anf_nodes.end(), item_use_node.first) ==
                fusion_info.anf_nodes.end()) {
              (*buffer_fusion_infos)[fusion_id].outputs_list.push_back(getitem);
              break;
            }
          }
        }
      }
    }
  }
}

void SetFusionOpRefInfos(session::KernelGraph *kernel_graph, const std::vector<AnfNodePtr> &outputs_list,
                         const AnfNodePtr &fusion_kernel) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (size_t idx = 0; idx < outputs_list.size(); ++idx) {
    auto output = outputs_list[idx];
    if (output->isa<CNode>() && AnfAlgo::GetCNodeName(output) == prim::kPrimTupleGetItem->name()) {
      auto real_output = AnfAlgo::VisitKernel(output, 0);
      auto output_cnode = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      auto input2 = output_cnode->input(2);
      auto output_idx = GetValue<int>(GetValueNode(input2));
      session::AnfWithOutIndex out_pair(real_output.first, output_idx);
      if (kernel_graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = kernel_graph->GetRefCorrespondOutput(out_pair);
        session::AnfWithOutIndex fusion_final_pair(fusion_kernel, idx);
        kernel_graph->AddRefCorrespondPairs(fusion_final_pair, origin_pair);
      }
    } else {
      session::AnfWithOutIndex out_pair(output, 0);
      if (kernel_graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = kernel_graph->GetRefCorrespondOutput(out_pair);
        session::AnfWithOutIndex fusion_final_pair(fusion_kernel, idx);
        kernel_graph->AddRefCorrespondPairs(fusion_final_pair, origin_pair);
      }
    }
  }
}
}  // namespace

void BufferFusion::SetRecordFusionId(const std::unordered_set<AnfNodePtr> &record) {
  auto id = fusion_id_allocator.AllocateFusionId();
  for (auto node : record) {
    fusion_id_allocator.SetFusionId(node, id);
  }
}

void BufferFusion::MatchConvBnreduce(const CNodePtr &cnode, const session::KernelGraph &kernel_graph,
                                     FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto conv = cnode->input(1);
  if (conv->isa<CNode>() && AnfAlgo::GetCNodeName(conv) == prim::kPrimConv2D->name()) {
    std::vector<int> output_used_num{SizeToInt(manager->node_users()[conv].size())};
    AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), conv);
    std::unordered_set<AnfNodePtr> record{cnode, conv};
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void BufferFusion::MatchBnupdateRelu(const CNodePtr &cnode, const AnfNodePtr &relu_input,
                                     const session::KernelGraph &kernel_graph, FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto getitem = relu_input->cast<CNodePtr>();
  auto bnupdate = getitem->input(1);
  if (bnupdate->isa<CNode>() && AnfAlgo::GetCNodeName(bnupdate) == kBNTrainingUpdateOpName) {
    std::vector<int> output_used_num(AnfAlgo::GetOutputTensorNum(bnupdate), 0);
    for (auto out_getitem : manager->node_users()[bnupdate]) {
      auto out_getitem_ptr = out_getitem.first->cast<CNodePtr>();
      auto input2 = out_getitem_ptr->input(2);
      auto output_idx = GetValue<int>(GetValueNode(input2));
      output_used_num[output_idx] = SizeToInt(manager->node_users()[out_getitem.first].size());
    }
    AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), bnupdate);
    std::unordered_set<AnfNodePtr> record{cnode, bnupdate};
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void BufferFusion::MatchBnupdateAddRelu(const CNodePtr &cnode, const AnfNodePtr &relu_input,
                                        const session::KernelGraph &kernel_graph, FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto add = relu_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add);
  auto tuple_getitem = add->input(1);
  if (tuple_getitem->isa<CNode>() && AnfAlgo::GetCNodeName(tuple_getitem) == prim::kPrimTupleGetItem->name()) {
    auto getitem = tuple_getitem->cast<CNodePtr>();
    auto bnupdate = getitem->input(1);
    if (bnupdate->isa<CNode>() && AnfAlgo::GetCNodeName(bnupdate) == kBNTrainingUpdateOpName) {
      std::vector<int> output_used_num(AnfAlgo::GetOutputTensorNum(bnupdate), 0);
      for (auto out_getitem : manager->node_users()[bnupdate]) {
        auto out_getitem_ptr = out_getitem.first->cast<CNodePtr>();
        auto input2 = out_getitem_ptr->input(2);
        auto output_idx = GetValue<int>(GetValueNode(input2));
        output_used_num[output_idx] = SizeToInt(manager->node_users()[out_getitem.first].size());
      }
      AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), bnupdate);
      std::unordered_set<AnfNodePtr> record{cnode, relu_input, bnupdate};
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
  }
}

void BufferFusion::MatchDepthwiseConvRelu(const CNodePtr &cnode, const session::KernelGraph &kernel_graph,
                                          FusedNodeRecord *candidate_fusion, bool is_order) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (is_order) {
    // DepthwiseConvolution--->Elemwise
    auto depthwise_conv = cnode->input(1);
    MS_EXCEPTION_IF_NULL(depthwise_conv);
    if (cnode->isa<CNode>() && AnfAlgo::GetCNodeName(depthwise_conv) == prim::kPrimDepthwiseConv2dNative->name()) {
      std::vector<int> output_used_num{SizeToInt(manager->node_users()[depthwise_conv].size())};
      AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), depthwise_conv);
      std::unordered_set<AnfNodePtr> record{cnode, depthwise_conv};
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
  } else {
    // Elemwise-->DepthwiseConvolution
    auto relu = cnode->input(1);
    MS_EXCEPTION_IF_NULL(relu);
    if (cnode->isa<CNode>() &&
        (AnfAlgo::GetCNodeName(relu) == prim::kPrimRelu->name() || AnfAlgo::GetCNodeName(relu) == kReluV2OpName)) {
      std::vector<int> output_used_num{SizeToInt(manager->node_users()[relu].size())};
      AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), relu);
      std::unordered_set<AnfNodePtr> record{cnode, relu};
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
  }
}

void BufferFusion::MatchOpNamePattern(const session::KernelGraph &kernel_graph, FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    if (!AnfAlgo::IsRealCNodeKernel(node) || fusion_id_allocator.HasFusionIdAttr(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::GetCNodeName(cnode) == kBNTrainingReduceOpName) {
      MatchConvBnreduce(cnode, kernel_graph, candidate_fusion);
    } else if (AnfAlgo::GetCNodeName(cnode) == kReluV2OpName ||
               AnfAlgo::GetCNodeName(cnode) == prim::kPrimRelu->name()) {
      auto relu_input = cnode->input(1);
      if (relu_input->isa<CNode>() && AnfAlgo::GetCNodeName(relu_input) == prim::kPrimTensorAdd->name()) {
        MatchBnupdateAddRelu(cnode, relu_input, kernel_graph, candidate_fusion);
      } else if (relu_input->isa<CNode>() && AnfAlgo::GetCNodeName(relu_input) == prim::kPrimTupleGetItem->name()) {
        MatchBnupdateRelu(cnode, relu_input, kernel_graph, candidate_fusion);
      } else if (relu_input->isa<CNode>() &&
                 AnfAlgo::GetCNodeName(relu_input) == prim::kPrimDepthwiseConv2dNative->name()) {
        MatchDepthwiseConvRelu(cnode, kernel_graph, candidate_fusion, true);
      }
    } else if (AnfAlgo::GetCNodeName(cnode) == prim::kPrimDepthwiseConv2dNative->name()) {
      MatchDepthwiseConvRelu(cnode, kernel_graph, candidate_fusion, false);
    }
  }
}

void BufferFusion::MatchFusionTypePattern(const session::KernelGraph &kernel_graph, FusedNodeRecord *candidate_fusion) {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(candidate_fusion);

  auto return_node = kernel_graph.get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->inputs().size() <= 1) {
    return;
  }
  std::deque<AnfNodePtr> todo;
  todo.push_back(return_node->input(1));
  std::unordered_set<AnfNodePtr> visited_set;

  while (!todo.empty()) {
    auto node = todo.front();
    MS_EXCEPTION_IF_NULL(node);
    todo.pop_front();
    std::unordered_set<AnfNodePtr> record;
    if (visited_set.find(node) != visited_set.end() || fusion_id_allocator.HasFusionIdAttr(node)) {
      continue;
    }
    // Only fuse real cnode
    if (!AnfAlgo::IsRealCNodeKernel(node)) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode != nullptr) {
        (void)todo.insert(todo.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
      }
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // cnode maybe updated
    cnode = FindFusionAnfNode(manager.get(), &visited_set, &record, &todo, cnode);
    if (record.size() >= MIN_PATTERN_SIZE && record.size() <= MAX_PATTERN_SIZE) {
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
    if (record.find(cnode) == record.end()) {
      todo.push_back(cnode);
    }
    // no node matched
    if (record.size() == 0) {
      (void)visited_set.insert(node);
    }
    (void)todo.insert(todo.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  }
}

void BufferFusion::GetBufferFusionInfo(session::KernelGraph *kernel_graph,
                                       std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) const {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  GetFusionScopeComputeNodeList(kernel_graph, buffer_fusion_infos);
  GetFusionScopeInputNodeList(*kernel_graph, buffer_fusion_infos);
  GetFusionScopeOutputNodeList(kernel_graph, buffer_fusion_infos);
  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    buffer_fusion_info.second.kernel_build_info =
      CreateFusionOpKernelInfo(buffer_fusion_info.second.inputs_list, buffer_fusion_info.second.outputs_list);
  }
}

bool BufferFusion::FuseBufferFusionPattern(session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool change = false;
  std::unordered_map<int32_t, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion_infos.clear();
  GetBufferFusionInfo(kernel_graph, &buffer_fusion_infos);

  std::vector<mindspore::kernel::FusionScopeInfo> fusion_scope_infos;
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    mindspore::kernel::FusionScopeInfo fusion_scope_info;
    fusion_scope_info.scope_id = buffer_fusion_info.first;
    fusion_scope_info.input_nodes = buffer_fusion_info.second.inputs_list;
    fusion_scope_info.compute_nodes = buffer_fusion_info.second.anf_nodes;
    fusion_scope_info.output_nodes = buffer_fusion_info.second.outputs_list;
    fusion_scope_infos.push_back(fusion_scope_info);
#ifdef DEBUG
    DumpFusionScopeInfo(fusion_scope_info);
#endif
  }
  auto kernel_mods = mindspore::kernel::KernelFusion(fusion_scope_infos);

  std::vector<int32_t> fusion_ids;
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    MS_LOG(DEBUG) << "anf node size: " << buffer_fusion_info.second.anf_nodes.size()
                  << ", inputs_list size: " << buffer_fusion_info.second.inputs_list.size()
                  << ", outputs list size: " << buffer_fusion_info.second.outputs_list.size();
    fusion_ids.push_back(buffer_fusion_info.first);
  }
  // Replace fusion op from return to head
  std::sort(fusion_ids.begin(), fusion_ids.end());
  for (auto &fusion_id : fusion_ids) {
    // Get kernel mod when supporting tbe
    if (kernel_mods.find(fusion_id) == kernel_mods.end() || kernel_mods[fusion_id] == nullptr) {
      MS_LOG(DEBUG) << "fusion id: " << fusion_id << ", fusion op compiling failed";
      continue;
    }
    change = ReplaceFusionOp(&buffer_fusion_infos, fusion_id, kernel_mods[fusion_id], kernel_graph);
  }
  MS_LOG(DEBUG) << "End Buffer Fusion";
  return change;
}

bool BufferFusion::MatchBufferFusionPattern(const session::KernelGraph &kernel_graph) {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto return_node = kernel_graph.get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->inputs().size() <= 1) {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern start...";
  FusedNodeRecord candidate_fusion;

  MatchOpNamePattern(kernel_graph, &candidate_fusion);
  MatchFusionTypePattern(kernel_graph, &candidate_fusion);

  if (candidate_fusion.empty()) {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern Success...";
  return true;
}

bool BufferFusion::ReplaceFusionOp(std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos,
                                   int32_t fusion_id, const kernel::KernelModPtr &kernel_ptr,
                                   session::KernelGraph *kernel_graph) const {
  auto buffer_fusion_info = (*buffer_fusion_infos)[fusion_id];
  auto buffer_fusion = CreateFusionOp(buffer_fusion_info.inputs_list, buffer_fusion_info.outputs_list,
                                      buffer_fusion_info.anf_nodes, kernel_graph);
  AnfAlgo::SetSelectKernelBuildInfo(buffer_fusion_info.kernel_build_info, buffer_fusion.get());
  // Set abstract of fusion_op node
  std::vector<TypeId> types;
  std::vector<std::vector<size_t>> shapes;
  for (const auto &out_node : buffer_fusion_info.outputs_list) {
    for (size_t idx = 0; idx < AnfAlgo::GetOutputTensorNum(out_node); ++idx) {
      types.push_back(AnfAlgo::GetOutputInferDataType(out_node, idx));
      shapes.push_back(AnfAlgo::GetOutputInferShape(out_node, idx));
    }
  }
  if (types.empty() || shapes.empty()) {
    MS_LOG(WARNING) << "buffer_fusion_info.outputs_list is empty";
    return false;
  }
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, buffer_fusion.get());
  AnfAlgo::SetKernelMod(kernel_ptr, buffer_fusion.get());
  SetFusionOpRefInfos(kernel_graph, buffer_fusion_info.outputs_list, buffer_fusion);
  ReplaceOldNode(buffer_fusion_infos, fusion_id, buffer_fusion, kernel_graph);
  return true;
}

bool BufferFusion::Run(const FuncGraphPtr &graph) {
  bool changed = false;
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  fusion_id_allocator.Init();
  if (MatchBufferFusionPattern(*kernel_graph)) {
    changed = FuseBufferFusionPattern(kernel_graph.get());
  }
  // clear fusion_id attr
  for (auto &node : graph->nodes()) {
    if (node != nullptr && node->isa<CNode>()) {
      AnfAlgo::EraseNodeAttr(kAttrFusionId, node);
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
