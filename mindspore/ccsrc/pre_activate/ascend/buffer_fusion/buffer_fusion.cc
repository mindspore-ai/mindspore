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
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <memory>
#include <string>
#include <algorithm>

#include "kernel/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"
#include "device/kernel_info.h"
#include "utils/context/ms_context.h"

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

void SetAnfNodeFusionId(const FusedNodeRecord &record_node) {
  MS_LOG(DEBUG) << "Size of opt vector to be fused is " << record_node.size();
  int32_t id = 1;
  for (auto &record : record_node) {
    MS_LOG(DEBUG) << "No" << id << ", opt vector to be fused contain " << record.size() << " opt.";
    for (const auto &candidate : record) {
      ValuePtr fusion_id_v = MakeValue(id);
      AnfAlgo::SetNodeAttr(kOpAttrFusionId, fusion_id_v, candidate);
      MS_LOG(DEBUG) << "No " << id << ": " << candidate->DebugString();
    }
    id++;
  }
}

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

kernel::KernelBuildInfoPtr CreateFusionOpKernelInfo(const std::vector<AnfNodePtr> &inputs_list_in,
                                                    const std::vector<AnfNodePtr> &inputs_list,
                                                    const std::vector<AnfNodePtr> &outputs_list) {
  MS_LOG(DEBUG) << "Start Create Kernel Info";
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  // inputs format and data type
  std::vector<std::string> inputs_format;
  std::vector<TypeId> inputs_data_type;
  for (auto node : inputs_list_in) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    for (size_t input_index = 1; input_index < inputs.size(); ++input_index) {
      if (std::find(inputs_list.begin(), inputs_list.end(), inputs[input_index]) != inputs_list.end()) {
        inputs_format.push_back(AnfAlgo::GetInputFormat(node, input_index - 1));
        inputs_data_type.push_back(AnfAlgo::GetInputDeviceDataType(node, input_index - 1));
      }
    }
  }
  // outputs format and data type
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_data_type;
  for (size_t index = 0; index < outputs_list.size(); ++index) {
    for (size_t idx = 0; idx < AnfAlgo::GetOutputTensorNum(outputs_list[index]); ++idx) {
      auto kernel_with_index = AnfAlgo::VisitKernel(outputs_list[index], idx);
      outputs_format.push_back(AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
      outputs_data_type.push_back(AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second));
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

void ReplaceOldNode(const std::vector<AnfNodePtr> &outputs_list, const AnfNodePtr &buffer_fusion_kernel,
                    session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (outputs_list.size() == 1) {  // single output
    (void)manager->Replace(outputs_list[0], buffer_fusion_kernel);
  } else {  // multiple output
    size_t real_idx = 0;
    for (size_t index = 0; index < outputs_list.size(); ++index) {
      if (AnfAlgo::GetOutputTensorNum(outputs_list[index]) == 1) {
        auto tuple_item = CreateTupleGetItem(buffer_fusion_kernel, kernel_graph, real_idx++);
        (void)manager->Replace(outputs_list[index], tuple_item);
      } else {
        std::vector<AnfNodePtr> make_tuple_inputs;
        AbstractBasePtrList abstract_list;
        make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
        for (size_t idx = 0; idx < AnfAlgo::GetOutputTensorNum(outputs_list[index]); ++idx) {
          auto tuple_item = CreateTupleGetItem(buffer_fusion_kernel, kernel_graph, real_idx++);
          abstract_list.push_back(tuple_item->abstract());
          make_tuple_inputs.push_back(tuple_item);
        }
        AnfNodePtr make_tuple = kernel_graph->NewCNode(make_tuple_inputs);
        make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
        (void)manager->Replace(outputs_list[index], make_tuple);
      }
    }
  }
}

void GetInputList(const CNodePtr &node, const int32_t cur_fusion_id, std::vector<AnfNodePtr> *inputs_list) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(inputs_list);
  auto &inputs = node->inputs();
  for (size_t input_index = 1; input_index < inputs.size(); ++input_index) {
    auto input = inputs[input_index];
    if (AnfAlgo::IsRealCNodeKernel(input)) {
      if (AnfAlgo::HasNodeAttr(kOpAttrFusionId, input)) {
        auto fusion_id = AnfAlgo::GetNodeAttr<int32_t>(input, kOpAttrFusionId);
        if (fusion_id != cur_fusion_id) {
          inputs_list->push_back(input);
        }
      } else {
        inputs_list->push_back(input);
      }
    } else if (input->isa<CNode>()) {
      for (auto &input_in : input->cast<CNodePtr>()->inputs()) {
        if (AnfAlgo::IsRealCNodeKernel(input_in)) {
          if (AnfAlgo::HasNodeAttr(kOpAttrFusionId, input_in)) {
            auto fusion_id = AnfAlgo::GetNodeAttr<int32_t>(input_in, kOpAttrFusionId);
            if (fusion_id != cur_fusion_id) {
              inputs_list->push_back(input);
            }
          } else {
            inputs_list->push_back(input);
          }
        }
      }
    } else {
      inputs_list->push_back(input);
    }
  }
}

void CheckCurrentNodeIsInput(const CNodePtr &node, const int32_t &cur_fusion_id,
                             std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  if ((*buffer_fusion_infos).find(cur_fusion_id) == (*buffer_fusion_infos).end()) {
    BufferFusionInfo_t buffer_fusion_info;
    (*buffer_fusion_infos)[cur_fusion_id] = buffer_fusion_info;
  }
  std::vector<AnfNodePtr> inputs_list;
  GetInputList(node, cur_fusion_id, &inputs_list);
  if (!inputs_list.empty()) {
    if (!(*buffer_fusion_infos)[cur_fusion_id].inputs_list.empty()) {
      (void)(*buffer_fusion_infos)[cur_fusion_id].inputs_list.insert(
        (*buffer_fusion_infos)[cur_fusion_id].inputs_list.end(), inputs_list.begin(), inputs_list.end());
      (void)(*buffer_fusion_infos)[cur_fusion_id].inputs_list_in.insert(
        (*buffer_fusion_infos)[cur_fusion_id].inputs_list_in.end(), node);
    } else {
      (*buffer_fusion_infos)[cur_fusion_id].inputs_list = inputs_list;
      (*buffer_fusion_infos)[cur_fusion_id].inputs_list_in.push_back(node);
    }
  }
}

void InsertNode(const AnfNodePtr &node, std::vector<AnfNodePtr> *list) {
  MS_EXCEPTION_IF_NULL(list);
  if (std::find(list->begin(), list->end(), node) == list->end()) {
    (void)list->insert(list->end(), node);
  }
}

void CheckCurrentNodeIsOutput(const CNodePtr &node, const int32_t &cur_fusion_id,
                              std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  for (auto &input : node->inputs()) {
    MS_EXCEPTION_IF_NULL(input);
    if (AnfAlgo::IsRealCNodeKernel(input) && AnfAlgo::HasNodeAttr(kOpAttrFusionId, input)) {
      auto fusion_id = AnfAlgo::GetNodeAttr<int32_t>(input, kOpAttrFusionId);
      if (buffer_fusion_infos->find(fusion_id) == buffer_fusion_infos->end()) {
        BufferFusionInfo_t buffer_fusion_info;
        (*buffer_fusion_infos)[fusion_id] = buffer_fusion_info;
      }
      if (fusion_id != cur_fusion_id) {
        InsertNode(input, &((*buffer_fusion_infos)[fusion_id].outputs_list));
      }
    } else if (input->isa<CNode>()) {
      for (auto &input_in : input->cast<CNodePtr>()->inputs()) {
        if (AnfAlgo::IsRealCNodeKernel(input_in) && AnfAlgo::HasNodeAttr(kOpAttrFusionId, input_in)) {
          auto fusion_id = AnfAlgo::GetNodeAttr<int32_t>(input_in, kOpAttrFusionId);
          if (buffer_fusion_infos->find(fusion_id) == buffer_fusion_infos->end()) {
            BufferFusionInfo_t buffer_fusion_info;
            (*buffer_fusion_infos)[fusion_id] = buffer_fusion_info;
          }
          if (fusion_id != cur_fusion_id) {
            InsertNode(input_in, &((*buffer_fusion_infos)[fusion_id].outputs_list));
          }
        }
      }
    }
  }
}

void GetFusionScopeNodeList(const session::KernelGraph &kernel_graph,
                            std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto nodes = TopoSort(kernel_graph.get_return());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::IsRealCNodeKernel(node) && AnfAlgo::HasNodeAttr(kOpAttrFusionId, node)) {
      auto fusion_id = AnfAlgo::GetNodeAttr<int32_t>(node, kOpAttrFusionId);
      (*buffer_fusion_infos)[fusion_id].anf_nodes.push_back(node);
    }
  }
}

void MatchOpNamePattern(const session::KernelGraph &kernel_graph, std::unordered_set<AnfNodePtr> *fused_set,
                        FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(fused_set);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    if (!AnfAlgo::IsRealCNodeKernel(node) || fused_set->find(node) != fused_set->end()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::GetCNodeName(cnode) == kBNTrainingReduceOpName) {
      auto conv = cnode->input(1);
      if (conv->isa<CNode>() && AnfAlgo::GetCNodeName(conv) == prim::kPrimConv2D->name()) {
        auto manager = kernel_graph.manager();
        MS_EXCEPTION_IF_NULL(manager);
        auto &users = manager->node_users();
        AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(users[conv].size()), conv);
        std::unordered_set<AnfNodePtr> record({cnode, conv});
        candidate_fusion->push_back(record);
        fused_set->insert(record.begin(), record.end());
      }
    }
  }
}

void MatchFusionTypePattern(const session::KernelGraph &kernel_graph, std::unordered_set<AnfNodePtr> *fused_set,
                            FusedNodeRecord *candidate_fusion) {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(fused_set);
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
    if (visited_set.find(node) != visited_set.end() || fused_set->find(node) != fused_set->end()) {
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
      fused_set->insert(record.begin(), record.end());
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
}  // namespace

void BufferFusion::GetBufferFusionInfo(const session::KernelGraph &kernel_graph,
                                       std::unordered_map<int32_t, BufferFusionInfo_t> *buffer_fusion_infos) const {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    if (!AnfAlgo::IsRealCNodeKernel(node)) {
      continue;
    }

    int32_t cur_fusion_id = -1;
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::HasNodeAttr(kOpAttrFusionId, cnode)) {
      cur_fusion_id = AnfAlgo::GetNodeAttr<int32_t>(cnode, kOpAttrFusionId);
      CheckCurrentNodeIsInput(cnode, cur_fusion_id, buffer_fusion_infos);
    }
    // Check if current node is output
    CheckCurrentNodeIsOutput(cnode, cur_fusion_id, buffer_fusion_infos);
  }

  GetFusionScopeNodeList(kernel_graph, buffer_fusion_infos);
  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    buffer_fusion_info.second.kernel_build_info =
      CreateFusionOpKernelInfo(buffer_fusion_info.second.inputs_list_in, buffer_fusion_info.second.inputs_list,
                               buffer_fusion_info.second.outputs_list);
  }
}

bool BufferFusion::FuseBufferFusionPattern(session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool change = false;
  std::unordered_map<int32_t, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion_infos.clear();
  GetBufferFusionInfo(*kernel_graph, &buffer_fusion_infos);

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
    change = ReplaceFusionOp(buffer_fusion_infos[fusion_id], kernel_mods[fusion_id], kernel_graph);
  }
  MS_LOG(DEBUG) << "End Buffer Fusion";
  return change;
}

bool BufferFusion::MatchBufferFusionPattern(const session::KernelGraph &kernel_graph) const {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto return_node = kernel_graph.get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->inputs().size() <= 1) {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern start...";
  FusedNodeRecord candidate_fusion;
  std::unordered_set<AnfNodePtr> fused_set;

  MatchOpNamePattern(kernel_graph, &fused_set, &candidate_fusion);
  MatchFusionTypePattern(kernel_graph, &fused_set, &candidate_fusion);

  if (!candidate_fusion.empty()) {
    SetAnfNodeFusionId(candidate_fusion);
  } else {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern Success...";
  return true;
}

bool BufferFusion::ReplaceFusionOp(const BufferFusionInfo_t &buffer_fusion_info, const kernel::KernelModPtr &kernel_ptr,
                                   session::KernelGraph *kernel_graph) const {
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
  // replace node
  ReplaceOldNode(buffer_fusion_info.outputs_list, buffer_fusion, kernel_graph);
  return true;
}

bool BufferFusion::Run(const FuncGraphPtr &graph) {
  bool changed = false;
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  if (MatchBufferFusionPattern(*kernel_graph)) {
    changed = FuseBufferFusionPattern(kernel_graph.get());
  }
  // clear fusion_id attr
  for (auto &node : graph->nodes()) {
    if (node != nullptr && node->isa<CNode>()) {
      AnfAlgo::EraseNodeAttr(kOpAttrFusionId, node);
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
