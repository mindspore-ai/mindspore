/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/buffer_fusion/ub_pattern_fusion.h"
#include <vector>
#include <utility>
#include <unordered_map>
#include <deque>
#include <memory>
#include <string>
#include <algorithm>
#include "backend/kernel_compiler/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "runtime/device/kernel_info.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"

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
    inputs_format.emplace_back(AnfAlgo::GetOutputFormat(real_input.first, real_input.second));
    inputs_data_type.emplace_back(AnfAlgo::GetOutputDeviceDataType(real_input.first, real_input.second));
  }
  // outputs format and data type
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_data_type;
  for (const auto &output : outputs_list) {
    if (AnfAlgo::GetCNodeName(output) == prim::kPrimTupleGetItem->name()) {
      auto tuple_getitem = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_getitem);
      outputs_format.emplace_back(AnfAlgo::GetOutputFormat(
        tuple_getitem->input(1), LongToSize(GetValue<int64_t>(GetValueNode(tuple_getitem->input(2))))));
      outputs_data_type.emplace_back(AnfAlgo::GetOutputDeviceDataType(
        tuple_getitem->input(1), LongToSize(GetValue<int64_t>(GetValueNode(tuple_getitem->input(2))))));
    } else {
      outputs_format.emplace_back(AnfAlgo::GetOutputFormat(output, 0));
      outputs_data_type.emplace_back(AnfAlgo::GetOutputDeviceDataType(output, 0));
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
  auto idx = NewValueNode(SizeToLong(output_index));
  MS_EXCEPTION_IF_NULL(idx);
  int64_t temp = SizeToLong(output_index);
  auto imm = std::make_shared<Int64Imm>(temp);
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

void ReplaceInputNodeInOtherFusionScope(std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos,
                                        int64_t fusion_id, const AnfNodePtr &output_item,
                                        const AnfNodePtr &replace_item) {
  for (int64_t id = fusion_id + 1; id <= SizeToLong(buffer_fusion_infos->size()); ++id) {
    auto itr = std::find((*buffer_fusion_infos)[id].inputs_list.begin(), (*buffer_fusion_infos)[id].inputs_list.end(),
                         output_item);
    if (itr != (*buffer_fusion_infos)[id].inputs_list.end()) {
      MS_LOG(DEBUG) << "replace input of other pattern, id = " << id;
      *itr = replace_item;
    }
  }
}

void ReplaceOldNode(std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos, int64_t fusion_id,
                    const AnfNodePtr &buffer_fusion_kernel, session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto buffer_fusion_info = (*buffer_fusion_infos)[fusion_id];
  if (buffer_fusion_info.outputs_list.size() == 1) {  // single output
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(buffer_fusion_info.outputs_list[0], buffer_fusion_kernel);
    }
    (void)manager->Replace(buffer_fusion_info.outputs_list[0], buffer_fusion_kernel);
    ReplaceInputNodeInOtherFusionScope(buffer_fusion_infos, fusion_id, buffer_fusion_info.outputs_list[0],
                                       buffer_fusion_kernel);
  } else {  // multiple output
    for (size_t index = 0; index < buffer_fusion_info.outputs_list.size(); ++index) {
      auto tuple_item = CreateTupleGetItem(buffer_fusion_kernel, kernel_graph, index);
      if (kernel_graph != nullptr) {
        kernel_graph->FrontBackendlMapUpdate(buffer_fusion_info.outputs_list[index], tuple_item);
      }
      (void)manager->Replace(buffer_fusion_info.outputs_list[index], tuple_item);
      ReplaceInputNodeInOtherFusionScope(buffer_fusion_infos, fusion_id, buffer_fusion_info.outputs_list[index],
                                         tuple_item);
    }
  }
}

void GetFusionScopeComputeNodeList(session::KernelGraph *kernel_graph,
                                   std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = TopoSort(kernel_graph->get_return());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (AnfAlgo::IsRealCNodeKernel(cnode) && AnfAlgo::HasNodeAttr(kOpAttrFusionId, cnode)) {
      auto fusion_id = AnfAlgo::GetNodeAttr<int64_t>(cnode, kOpAttrFusionId);
      (*buffer_fusion_infos)[fusion_id].anf_nodes.push_back(cnode);
    }
  }
}

void GetFusionScopeInputNodeList(const session::KernelGraph &kernel_graph,
                                 std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    auto fusion_id = buffer_fusion_info.first;
    const auto &fusion_info = buffer_fusion_info.second;
    for (const auto &node : fusion_info.anf_nodes) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      for (size_t idx = 1; idx < cnode->inputs().size(); ++idx) {
        auto real_input = AnfAlgo::VisitKernel(cnode->input(idx), 0);
        if (std::find(fusion_info.anf_nodes.begin(), fusion_info.anf_nodes.end(), real_input.first) ==
            fusion_info.anf_nodes.end()) {
          if (auto in = cnode->input(idx); std::find((*buffer_fusion_infos)[fusion_id].inputs_list.begin(),
                                                     (*buffer_fusion_infos)[fusion_id].inputs_list.end(),
                                                     in) == (*buffer_fusion_infos)[fusion_id].inputs_list.end()) {
            if (!HasAbstractMonad(in)) {
              (*buffer_fusion_infos)[fusion_id].inputs_list.push_back(in);
            }
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
  if (getitem1->size() < kTupleGetItemInputSize) {
    MS_LOG(EXCEPTION) << "node's input size less than " << kTupleGetItemInputSize << ", getitem1["
                      << getitem1->DebugString() << "]";
  }
  if (getitem2->size() < kTupleGetItemInputSize) {
    MS_LOG(EXCEPTION) << "node's input size less than " << kTupleGetItemInputSize << ", getitem1["
                      << getitem2->DebugString() << "]";
  }
  auto output_idx1 = GetValue<int64_t>(GetValueNode(getitem1->input(2)));
  auto output_idx2 = GetValue<int64_t>(GetValueNode(getitem2->input(2)));
  return output_idx1 < output_idx2;
}

void GetFusionScopeOutputNodeList(session::KernelGraph *kernel_graph,
                                  std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    auto fusion_id = buffer_fusion_info.first;
    const auto &fusion_info = buffer_fusion_info.second;
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
        int64_t prev_idx = 0;
        std::vector<AnfNodePtr> tuple_getitem_nodes;
        std::transform(manager->node_users()[node].begin(), manager->node_users()[node].end(),
                       std::back_inserter(tuple_getitem_nodes),
                       [](const std::pair<AnfNodePtr, int> &use_node) { return use_node.first; });
        std::sort(tuple_getitem_nodes.begin(), tuple_getitem_nodes.end(), TupleGetitemNodeCompare);
        for (auto &getitem : tuple_getitem_nodes) {
          MS_EXCEPTION_IF_NULL(getitem);
          auto getitem_ptr = getitem->cast<CNodePtr>();
          auto input2 = getitem_ptr->input(2);
          auto output_idx = GetValue<int64_t>(GetValueNode(input2));
          for (int64_t stub_idx = prev_idx; stub_idx < output_idx; ++stub_idx) {
            auto stub_node = CreateTupleGetItem(node, kernel_graph, LongToSize(stub_idx));
            (*buffer_fusion_infos)[fusion_id].outputs_list.push_back(stub_node);
          }
          prev_idx = output_idx + 1;
          for (auto &item_use_node : manager->node_users()[getitem]) {
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
    MS_EXCEPTION_IF_NULL(output);
    if (output->isa<CNode>() && AnfAlgo::GetCNodeName(output) == prim::kPrimTupleGetItem->name()) {
      auto real_output = AnfAlgo::VisitKernel(output, 0);
      auto output_cnode = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      auto input2 = output_cnode->input(2);
      auto output_idx = GetValue<int64_t>(GetValueNode(input2));
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

void RemoveCircle(const session::KernelGraph &kernel_graph,
                  std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  std::vector<int64_t> fusion_ids;
  for (auto &[fusion_id, fusion_info] : *buffer_fusion_infos) {
    bool has_circle = false;
    for (auto &inp : fusion_info.inputs_list) {
      MS_EXCEPTION_IF_NULL(inp);
      if (!inp->isa<CNode>() || AnfAlgo::CheckPrimitiveType(inp, prim::kPrimLoad)) {
        continue;
      }

      if (IsDepend(kernel_graph, inp, fusion_info.anf_nodes)) {
        has_circle = true;
        break;
      }
    }

    if (has_circle) {
      fusion_ids.emplace_back(fusion_id);
    }
  }

  for (auto &fusion_id : fusion_ids) {
    buffer_fusion_infos->erase(fusion_id);
  }
}
}  // namespace

void UbPatternFusion::GetBufferFusionInfo(session::KernelGraph *kernel_graph,
                                          std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) const {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  GetFusionScopeComputeNodeList(kernel_graph, buffer_fusion_infos);
  GetFusionScopeInputNodeList(*kernel_graph, buffer_fusion_infos);
  GetFusionScopeOutputNodeList(kernel_graph, buffer_fusion_infos);
  // Remove the fusion infos which will produce a circle if do fusion
  RemoveCircle(*kernel_graph, buffer_fusion_infos);

  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    buffer_fusion_info.second.kernel_build_info =
      CreateFusionOpKernelInfo(buffer_fusion_info.second.inputs_list, buffer_fusion_info.second.outputs_list);
    // just for full_name_with_scope for every buffer_fusion_info.
    auto fusion_node = CreateFusionOp(buffer_fusion_info.second.inputs_list, buffer_fusion_info.second.outputs_list,
                                      buffer_fusion_info.second.anf_nodes, kernel_graph);
    MS_EXCEPTION_IF_NULL(fusion_node);
    buffer_fusion_info.second.full_name = fusion_node->fullname_with_scope();
  }
}

bool UbPatternFusion::FuseBufferFusionPattern(session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool change = false;
  std::unordered_map<int64_t, BufferFusionInfo_t> buffer_fusion_infos;
  GetBufferFusionInfo(kernel_graph, &buffer_fusion_infos);

  std::vector<mindspore::kernel::FusionScopeInfo> fusion_scope_infos;
  std::transform(
    buffer_fusion_infos.begin(), buffer_fusion_infos.end(), std::back_inserter(fusion_scope_infos),
    [](const std::pair<int64_t, BufferFusionInfo_t> &buffer_fusion_info) -> mindspore::kernel::FusionScopeInfo {
      return mindspore::kernel::FusionScopeInfo(
        buffer_fusion_info.first, buffer_fusion_info.second.full_name, buffer_fusion_info.second.inputs_list,
        buffer_fusion_info.second.anf_nodes, buffer_fusion_info.second.outputs_list);
    });
  auto kernel_mods = mindspore::kernel::KernelFusion(fusion_scope_infos);
  std::set<int64_t> fusion_ids;
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    MS_LOG(DEBUG) << "anf node size: " << buffer_fusion_info.second.anf_nodes.size()
                  << ", inputs_list size: " << buffer_fusion_info.second.inputs_list.size()
                  << ", outputs list size: " << buffer_fusion_info.second.outputs_list.size();
    fusion_ids.insert(buffer_fusion_info.first);
  }
  // Replace fusion op from return to head
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

bool UbPatternFusion::ReplaceFusionOp(std::unordered_map<int64_t, BufferFusionInfo_t> *buffer_fusion_infos,
                                      int64_t fusion_id, const kernel::KernelModPtr &kernel_ptr,
                                      session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto buffer_fusion_info = (*buffer_fusion_infos)[fusion_id];
  if (buffer_fusion_info.anf_nodes.size() < 2) {
    return false;
  }
  TraceGuard guard(std::make_shared<TraceOpt>(buffer_fusion_info.anf_nodes[0]->debug_info()));
  auto buffer_fusion = CreateFusionOp(buffer_fusion_info.inputs_list, buffer_fusion_info.outputs_list,
                                      buffer_fusion_info.anf_nodes, kernel_graph);
  buffer_fusion->set_fullname_with_scope(buffer_fusion_info.full_name);
  AnfAlgo::SetSelectKernelBuildInfo(buffer_fusion_info.kernel_build_info, buffer_fusion.get());
  // Set abstract of fusion_op node
  std::vector<TypeId> types;
  std::vector<std::vector<size_t>> shapes;
  for (const auto &out_node : buffer_fusion_info.outputs_list) {
    size_t out_num = AnfAlgo::GetOutputTensorNum(out_node);
    for (size_t idx = 0; idx < out_num; ++idx) {
      types.emplace_back(AnfAlgo::GetOutputInferDataType(out_node, idx));
      shapes.emplace_back(AnfAlgo::GetOutputInferShape(out_node, idx));
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

bool UbPatternFusion::Run(const FuncGraphPtr &graph) {
  bool changed = false;
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  changed = FuseBufferFusionPattern(kernel_graph.get());
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
