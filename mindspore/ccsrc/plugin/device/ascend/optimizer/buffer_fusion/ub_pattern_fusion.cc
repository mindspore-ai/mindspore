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
#include "plugin/device/ascend/optimizer/buffer_fusion/ub_pattern_fusion.h"
#include <vector>
#include <utility>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <algorithm>
#include <iterator>
#include <list>

#include "utils/hash_map.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"
#include "backend/common/optimizer/helper.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/session/kernel_graph.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/robin_hood.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "ir/manager.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "kernel/kernel.h"
#include "kernel/kernel_build_info.h"
#include "kernel/kernel_fusion.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "utils/anf_utils.h"
#include "utils/compact_set.h"
#include "utils/convert_utils_base.h"
#include "utils/hash_set.h"
#include "utils/info.h"
#include "utils/log_adapter.h"
#include "utils/trace_info.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"

namespace mindspore {
namespace kernel {
namespace tbe {
class TbeUtils;
}  // namespace tbe
}  // namespace kernel

namespace opt {
using mindspore::kernel::tbe::TbeUtils;
namespace {
constexpr int8_t MAX_PATTERN_SIZE = 7;
constexpr int8_t MIN_PATTERN_SIZE = 2;
constexpr int8_t ELTWISE_INPUT_SIZE = 2;
constexpr int8_t ELTWISE_USE = 1;
constexpr int8_t MULTI_ELTWISE_USE = 2;
constexpr int8_t MAX_MULTI_ELTWISE_SIZE = 4;
constexpr int8_t MAX_PURE_BUFFER_SUCC_SIZE = 3;
constexpr size_t kFusionNodeNumThreshold = 2;
constexpr auto kOpAttrFusionId = "fusion_id";

#ifdef DEBUG
void DumpFusionScopeInfo(const kernel::FusionScopeInfo &info) {
  MS_LOG(INFO) << "=== Dump FusionScopeInfo start id: " << info.scope_id;
  for (auto &node : info.input_nodes) {
    MS_LOG(INFO) << "=== Input: " << node->DebugString();
  }
  for (auto &node : info.output_nodes) {
    MS_LOG(INFO) << "=== Output: " << node->DebugString();
  }
  for (auto &node : info.compute_nodes) {
    MS_LOG(INFO) << "=== Compute: (" << node->DebugString() << ")-("
                 << mindspore::kernel::GetFusionNameByType(AnfAlgo::GetFusionType(node)) << ")";
  }
  MS_LOG(INFO) << "=== Dump FusionScopeInfo end";
}
#endif
CNodePtr CreateFusionOp(const std::vector<AnfNodePtr> &inputs_list, const std::vector<AnfNodePtr> &outputs_list,
                        const std::vector<AnfNodePtr> &anf_nodes, session::KernelGraph *kernel_graph) {
  MS_LOG(DEBUG) << "Start Create FusionOp Kernel";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::string fusion_op_name = "FusionOp";
  for (auto &node : anf_nodes) {
    fusion_op_name += '_' + common::AnfAlgo::GetCNodeName(node);
  }
  auto fusion_op = std::make_shared<Primitive>(fusion_op_name);
  MS_EXCEPTION_IF_NULL(fusion_op);

  std::vector<std::string> input_names;
  for (size_t i = 0; i < inputs_list.size(); i++) {
    (void)input_names.emplace_back("input" + std::to_string(i));
  }
  std::vector<std::string> output_names;
  for (size_t i = 0; i < outputs_list.size(); i++) {
    (void)output_names.emplace_back("output" + std::to_string(i));
  }

  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  fusion_op->set_attr("input_names", input_names_v);
  fusion_op->set_attr("output_names", output_names_v);
  for (auto &node : anf_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if (common::AnfAlgo::HasNodeAttr(kAttrFracZGroup, cnode)) {
      auto fracz_group = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrFracZGroup);
      fusion_op->set_attr(kAttrFracZGroup, MakeValue(fracz_group));
    }
    if (common::AnfAlgo::HasNodeAttr(kAttrDump, cnode)) {
      auto dump_flag = common::AnfAlgo::GetNodeAttr<string>(node, kAttrDump);
      fusion_op->set_attr(kAttrDump, MakeValue(dump_flag));
    }
  }
  std::vector<AnfNodePtr> fusion_inputs_list = inputs_list;
  auto value_node = std::make_shared<ValueNode>(fusion_op);
  (void)fusion_inputs_list.insert(fusion_inputs_list.begin(), value_node);
  auto buffer_fusion_kernel = kernel_graph->NewCNode(fusion_inputs_list);
  if (buffer_fusion_kernel == nullptr) {
    MS_LOG(EXCEPTION) << "New FusionOp kernel failed!";
  }
  buffer_fusion_kernel->set_scope((anf_nodes.back())->scope());
  buffer_fusion_kernel->AddFusedDebugInfoList(anf_nodes);

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
    auto real_input = common::AnfAlgo::VisitKernel(input, 0);
    (void)inputs_format.emplace_back(AnfAlgo::GetOutputFormat(real_input.first, real_input.second));
    (void)inputs_data_type.emplace_back(AnfAlgo::GetOutputDeviceDataType(real_input.first, real_input.second));
  }
  // outputs format and data type
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_data_type;
  for (const auto &output : outputs_list) {
    if (common::AnfAlgo::GetCNodeName(output) == prim::kPrimTupleGetItem->name()) {
      auto tuple_getitem = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_getitem);
      (void)outputs_format.emplace_back(AnfAlgo::GetOutputFormat(
        tuple_getitem->input(kIndex1), LongToSize(GetValue<int64_t>(GetValueNode(tuple_getitem->input(kIndex2))))));
      (void)outputs_data_type.emplace_back(AnfAlgo::GetOutputDeviceDataType(
        tuple_getitem->input(kIndex1), LongToSize(GetValue<int64_t>(GetValueNode(tuple_getitem->input(kIndex2))))));
    } else {
      (void)outputs_format.emplace_back(AnfAlgo::GetOutputFormat(output, 0));
      (void)outputs_data_type.emplace_back(AnfAlgo::GetOutputDeviceDataType(output, 0));
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
  common::AnfAlgo::SetOutputTypeAndDetailShape(
    {common::AnfAlgo::GetOutputInferDataType(buffer_fusion_kernel, output_index)},
    {AnfAlgo::GetOutputDetailShape(buffer_fusion_kernel, output_index)}, tuple_item.get());
  return tuple_item;
}

void ReplaceInputNodeInOtherFusionScope(mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos,
                                        int64_t fusion_id, const AnfNodePtr &output_item,
                                        const AnfNodePtr &replace_item) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  for (int64_t id = fusion_id + 1; id <= SizeToLong(buffer_fusion_infos->size()); ++id) {
    auto itr = std::find((*buffer_fusion_infos)[id].inputs_list.begin(), (*buffer_fusion_infos)[id].inputs_list.end(),
                         output_item);
    if (itr != (*buffer_fusion_infos)[id].inputs_list.end()) {
      MS_LOG(DEBUG) << "Replace input of other pattern, id = " << id;
      *itr = replace_item;
    }
  }
}

void ReplaceOldNode(mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos, int64_t fusion_id,
                    const AnfNodePtr &buffer_fusion_kernel, session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto buffer_fusion_info = (*buffer_fusion_infos)[fusion_id];
  if (buffer_fusion_info.outputs_list.size() == 1) {  // single output
    kernel_graph->FrontBackendlMapUpdate(buffer_fusion_info.outputs_list[0], buffer_fusion_kernel);
    (void)manager->Replace(buffer_fusion_info.outputs_list[0], buffer_fusion_kernel);
    ReplaceInputNodeInOtherFusionScope(buffer_fusion_infos, fusion_id, buffer_fusion_info.outputs_list[0],
                                       buffer_fusion_kernel);
  } else {  // multiple output
    for (size_t index = 0; index < buffer_fusion_info.outputs_list.size(); ++index) {
      auto tuple_item = CreateTupleGetItem(buffer_fusion_kernel, kernel_graph, index);
      kernel_graph->FrontBackendlMapUpdate(buffer_fusion_info.outputs_list[index], tuple_item);
      (void)manager->Replace(buffer_fusion_info.outputs_list[index], tuple_item);
      ReplaceInputNodeInOtherFusionScope(buffer_fusion_infos, fusion_id, buffer_fusion_info.outputs_list[index],
                                         tuple_item);
    }
  }
}

void GetFusionScopeComputeNodeList(const session::KernelGraph *kernel_graph,
                                   mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = TopoSort(kernel_graph->get_return());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (AnfUtils::IsRealCNodeKernel(cnode) && common::AnfAlgo::HasNodeAttr(kOpAttrFusionId, cnode)) {
      auto fusion_id = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kOpAttrFusionId);
      (*buffer_fusion_infos)[fusion_id].anf_nodes.push_back(cnode);
    }
  }

  // If Graph Kernel Fusion is enabled, we will let Graph Kernel fuse these nodes if it supports.
  if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    auto iter = buffer_fusion_infos->begin();
    while (iter != buffer_fusion_infos->end()) {
      if (graphkernel::GraphKernelSupported(iter->second.anf_nodes)) {
        MS_LOG(DEBUG) << "Fusion id: " << iter->first << ", uses Graph Kernel Fusion";
        iter = buffer_fusion_infos->erase(iter);
      } else {
        (void)iter++;
      }
    }
  }
}

void GetFusionScopeInputNodeList(const session::KernelGraph &kernel_graph,
                                 mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    auto &fusion_info = buffer_fusion_info.second;
    fusion_info.all_inputs_to_first_node = true;

    for (size_t node_idx = 0; node_idx < fusion_info.anf_nodes.size(); ++node_idx) {
      const auto &node = fusion_info.anf_nodes[node_idx];
      MS_EXCEPTION_IF_NULL(node);
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      size_t old_input_num = fusion_info.inputs_list.size();
      for (size_t idx = 1; idx < cnode->inputs().size(); ++idx) {
        auto real_input = common::AnfAlgo::VisitKernel(cnode->input(idx), 0);
        if (std::find(fusion_info.anf_nodes.begin(), fusion_info.anf_nodes.end(), real_input.first) ==
            fusion_info.anf_nodes.end()) {
          if (!HasAbstractMonad(cnode->input(idx))) {
            fusion_info.inputs_list.push_back(cnode->input(idx));
            fusion_info.nodes_id.push_back(node_idx);
          }
        }
      }

      if (node_idx != 0 && fusion_info.inputs_list.size() != old_input_num) {
        fusion_info.all_inputs_to_first_node = false;
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
    MS_LOG(EXCEPTION) << "Node's input size less than " << kTupleGetItemInputSize << ", getitem1["
                      << getitem1->DebugString() << "]";
  }
  if (getitem2->size() < kTupleGetItemInputSize) {
    MS_LOG(EXCEPTION) << "Node's input size less than " << kTupleGetItemInputSize << ", getitem1["
                      << getitem2->DebugString() << "]";
  }
  auto output_idx1 = GetValue<int64_t>(GetValueNode(getitem1->input(kIndex2)));
  auto output_idx2 = GetValue<int64_t>(GetValueNode(getitem2->input(kIndex2)));
  return output_idx1 < output_idx2;
}

AnfNodePtr RemoveNodeFromUpdateState(session::KernelGraph *kernel_graph, const AnfNodePtr &node,
                                     const AnfNodePtr &updatestate) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(updatestate);
  auto updatestate_cnode = updatestate->cast<CNodePtr>();
  auto inputs = updatestate_cnode->inputs();
  std::vector<AnfNodePtr> new_inputs;
  (void)std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(new_inputs),
                     [node](const AnfNodePtr &input) { return node != input; });
  AnfNodePtr new_updatestate = nullptr;
  constexpr size_t updatestate_input_size = 3;
  // If there are only has one CNode in UpdateState's inputs
  // old_updatestate = UpdateState(umonad, cnode1)
  // cnode2 = CNode2(..., old_updatestate)
  // --> after remove the cnode1, mean that replace old_updatestate by umonad.
  // cnode2 = CNode2(..., umonad)
  if (new_inputs.size() < updatestate_input_size) {
    new_updatestate = updatestate_cnode->input(1);
  } else {
    new_updatestate = kernel_graph->NewCNode(new_inputs);
  }
  MS_EXCEPTION_IF_NULL(new_updatestate);
  new_updatestate->set_scope(updatestate->scope());
  new_updatestate->set_abstract(updatestate->abstract());
  return new_updatestate;
}

void GetFusionScopeOutputNodeList(session::KernelGraph *kernel_graph,
                                  mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &buffer_fusion_info : *buffer_fusion_infos) {
    auto &fusion_info = buffer_fusion_info.second;
    fusion_info.all_outputs_from_last_node = true;
    for (size_t node_idx = 0; node_idx < fusion_info.anf_nodes.size(); ++node_idx) {
      const auto &node = fusion_info.anf_nodes[node_idx];
      size_t old_output_num = fusion_info.outputs_list.size();
      if (AnfAlgo::GetOutputTensorNum(node) == 1) {
        auto use_nodes = manager->node_users()[node];
        for (const auto &use_node : use_nodes) {
          // Do not think of updatestate as real output,
          // Ensuring normal fusion requires eliminating the node of the updatestate
          if (common::AnfAlgo::CheckPrimitiveType(use_node.first, prim::kPrimUpdateState)) {
            auto new_updatestate = RemoveNodeFromUpdateState(kernel_graph, node, use_node.first);
            (void)manager->Replace(use_node.first, new_updatestate);
            continue;
          }
          if (std::find(fusion_info.anf_nodes.begin(), fusion_info.anf_nodes.end(), use_node.first) ==
              fusion_info.anf_nodes.end()) {
            fusion_info.outputs_list.push_back(node);
            break;
          }
        }
      } else {
        int64_t prev_idx = 0;
        std::vector<AnfNodePtr> tuple_getitem_nodes;
        auto users = manager->node_users()[node];
        for (auto &user : users) {
          if (common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimUpdateState)) {
            auto new_updatestate = RemoveNodeFromUpdateState(kernel_graph, node, user.first);
            (void)manager->Replace(user.first, new_updatestate);
            continue;
          }
          if (common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimTupleGetItem)) {
            (void)tuple_getitem_nodes.emplace_back(user.first);
          }
        }
        std::sort(tuple_getitem_nodes.begin(), tuple_getitem_nodes.end(), TupleGetitemNodeCompare);
        for (auto &getitem : tuple_getitem_nodes) {
          MS_EXCEPTION_IF_NULL(getitem);
          auto getitem_ptr = getitem->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(getitem_ptr);
          auto input2 = getitem_ptr->input(kIndex2);
          auto output_idx = GetValue<int64_t>(GetValueNode(input2));
          for (int64_t stub_idx = prev_idx; stub_idx < output_idx; ++stub_idx) {
            auto stub_node = CreateTupleGetItem(node, kernel_graph, LongToSize(stub_idx));
            fusion_info.outputs_list.push_back(stub_node);
          }
          prev_idx = output_idx + 1;
          for (auto &item_use_node : manager->node_users()[getitem]) {
            if (std::find(fusion_info.anf_nodes.begin(), fusion_info.anf_nodes.end(), item_use_node.first) ==
                fusion_info.anf_nodes.end()) {
              fusion_info.outputs_list.push_back(getitem);
              break;
            }
          }
        }
      }

      if (node_idx != fusion_info.anf_nodes.size() - 1 && fusion_info.outputs_list.size() != old_output_num) {
        fusion_info.all_outputs_from_last_node = false;
      }
    }
  }
}

void SetOutputUsedNumAttr(const session::KernelGraph &kernel_graph,
                          const mindspore::HashMap<int64_t, BufferFusionInfo_t> &buffer_fusion_infos) {
  for (auto &fusion_info : buffer_fusion_infos) {
    auto &fusion_nodes = fusion_info.second.anf_nodes;
    for (auto iter = fusion_nodes.begin(); iter != fusion_nodes.end() - 1; ++iter) {
      auto node = *iter;
      auto output_used_num = GetNodeOutputUsedNum(kernel_graph, node);
      common::AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), node);
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
    if (output->isa<CNode>() && common::AnfAlgo::GetCNodeName(output) == prim::kPrimTupleGetItem->name()) {
      auto real_output = common::AnfAlgo::VisitKernel(output, 0);
      auto output_cnode = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      auto input2 = output_cnode->input(kIndex2);
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

// As shown in the following graph, if A, B, and C are combined into E,
//  A -> B -> C
//    -> D ->
// then E and D form a cycle.
//    _
//  E _ D
bool CheckCircle(const session::KernelGraph &kernel_graph, const BufferFusionInfo_t &fusion_info) {
  MS_EXCEPTION_IF_CHECK_FAIL((fusion_info.inputs_list.size() == fusion_info.nodes_id.size()),
                             "Fusion_info size not equal");
  // The nodes do not form cycles before fusion. Checking is not necessary if one of the following conditions is met:
  // 1. All inputs to the fusion scope are passed to the first node in the scope.
  // 2. All outputs of the fusion scope are generated by the last node of the scope.
  if (fusion_info.all_inputs_to_first_node || fusion_info.all_outputs_from_last_node) {
    return false;
  }

  // Save visited nodes to avoid repeated visit and improve performance.
  mindspore::HashSet<AnfNodePtr> visited_nodes;

  bool has_circle = false;
  for (size_t i = 0; i < fusion_info.inputs_list.size(); i++) {
    const auto &inp = fusion_info.inputs_list.at(i);
    const auto &node_id = fusion_info.nodes_id.at(i);
    MS_EXCEPTION_IF_NULL(inp);
    // the inputs of the first node cannot access to the nodes in the fusion scope
    if (node_id == 0 || !inp->isa<CNode>() || common::AnfAlgo::CheckPrimitiveType(inp, prim::kPrimLoad)) {
      continue;
    }

    if (IsDepend(kernel_graph, inp, fusion_info.anf_nodes, &visited_nodes)) {
      has_circle = true;
      break;
    }
  }
  return has_circle;
}
}  // namespace

void UbPatternFusion::GetBufferFusionInfo(session::KernelGraph *kernel_graph,
                                          mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) const {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  GetFusionScopeComputeNodeList(kernel_graph, buffer_fusion_infos);
  GetFusionScopeInputNodeList(*kernel_graph, buffer_fusion_infos);
  GetFusionScopeOutputNodeList(kernel_graph, buffer_fusion_infos);
  SetOutputUsedNumAttr(*kernel_graph, *buffer_fusion_infos);

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
  mindspore::HashMap<int64_t, BufferFusionInfo_t> buffer_fusion_infos;
  GetBufferFusionInfo(kernel_graph, &buffer_fusion_infos);

  std::vector<mindspore::kernel::FusionScopeInfo> fusion_scope_infos;
  (void)std::transform(buffer_fusion_infos.begin(), buffer_fusion_infos.end(), std::back_inserter(fusion_scope_infos),
                       [](const auto &buffer_fusion_info) -> mindspore::kernel::FusionScopeInfo {
                         return mindspore::kernel::FusionScopeInfo(
                           buffer_fusion_info.first, buffer_fusion_info.second.full_name,
                           buffer_fusion_info.second.core_type, buffer_fusion_info.second.inputs_list,
                           buffer_fusion_info.second.anf_nodes, buffer_fusion_info.second.outputs_list);
                       });
  auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
  auto id_names = build_manager.TbeFusionOpCompile(fusion_scope_infos);
  std::set<int64_t> fusion_ids;
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    MS_LOG(DEBUG) << "Anf node size: " << buffer_fusion_info.second.anf_nodes.size()
                  << ", inputs_list size: " << buffer_fusion_info.second.inputs_list.size()
                  << ", outputs list size: " << buffer_fusion_info.second.outputs_list.size();
    fusion_ids.insert(buffer_fusion_info.first);
  }
  // Replace fusion op from return to head
  for (auto &fusion_id : fusion_ids) {
    // Get kernel mod when supporting tbe
    if (id_names.find(fusion_id) == id_names.end()) {
      MS_LOG(DEBUG) << "Fusion id: " << fusion_id << ", fusion op compiling failed";
      continue;
    }
    if (CheckCircle(*kernel_graph, buffer_fusion_infos[fusion_id])) {
      MS_LOG(DEBUG) << "Fusion id: " << fusion_id << " will cause graph circle, pass this fusion.";
    } else {
      change = ReplaceFusionOp(&buffer_fusion_infos, fusion_id, kernel_graph);
    }
  }
  MS_LOG(DEBUG) << "End Buffer Fusion";
  return change;
}

bool UbPatternFusion::ReplaceFusionOp(mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos,
                                      int64_t fusion_id, session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(buffer_fusion_infos);
  auto buffer_fusion_info = (*buffer_fusion_infos)[fusion_id];
  if (buffer_fusion_info.anf_nodes.size() < kFusionNodeNumThreshold) {
    return false;
  }
  TraceGuard guard(std::make_shared<TraceOpt>(buffer_fusion_info.anf_nodes[0]->debug_info()));
  auto buffer_fusion = CreateFusionOp(buffer_fusion_info.inputs_list, buffer_fusion_info.outputs_list,
                                      buffer_fusion_info.anf_nodes, kernel_graph);
  buffer_fusion->set_fullname_with_scope(buffer_fusion_info.full_name);
  AnfAlgo::SetSelectKernelBuildInfo(buffer_fusion_info.kernel_build_info, buffer_fusion.get());
  // Set abstract of fusion_op node
  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  for (const auto &out_node : buffer_fusion_info.outputs_list) {
    size_t out_num = AnfAlgo::GetOutputTensorNum(out_node);
    for (size_t idx = 0; idx < out_num; ++idx) {
      (void)types.emplace_back(common::AnfAlgo::GetOutputInferDataType(out_node, idx));
      (void)shapes.emplace_back(AnfAlgo::GetOutputDetailShape(out_node, idx));
    }
  }
  if (types.empty() || shapes.empty()) {
    MS_LOG(WARNING) << "The outputs_list of buffer_fusion_info is empty.";
    return false;
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, buffer_fusion.get());
  common::AnfAlgo::SetNodeAttr(kAttrIsUBFusionOp, MakeValue(true), buffer_fusion);
  SetFusionOpRefInfos(kernel_graph, buffer_fusion_info.outputs_list, buffer_fusion);
  ReplaceOldNode(buffer_fusion_infos, fusion_id, buffer_fusion, kernel_graph);
  return true;
}

bool UbPatternFusion::RunPass(const FuncGraphPtr &graph) {
  bool changed = false;
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  changed = FuseBufferFusionPattern(kernel_graph.get());
  // clear fusion_id attr
  for (auto &node : graph->nodes()) {
    if (node != nullptr && node->isa<CNode>()) {
      common::AnfAlgo::EraseNodeAttr(kAttrFusionId, node);
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
