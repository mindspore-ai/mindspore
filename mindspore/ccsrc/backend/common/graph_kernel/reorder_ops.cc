/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/reorder_ops.h"
#include <memory>
#include <vector>
#include <string>
#include "utils/hash_set.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "utils/log_adapter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
bool IsTypeInsensitive(const CNodePtr &node) {
  // Nodes that will change the input data type will not seen as type insensitive nodes.
  static mindspore::HashSet<PrimitivePtr> type_insensitive_op_list{
    prim::kPrimTransData, prim::kPrimTranspose, prim::kPrimTransposeD, prim::kPrimExpandDims, prim::kPrimReshape,
    prim::kPrimSqueeze,   prim::kPrimTile,      prim::kPrimNeg,        prim::kPrimReLU,       prim::kPrimRelu,
    prim::kPrimMaximum,   prim::kPrimMinimum,   prim::kPrimSelect};

  return std::any_of(type_insensitive_op_list.begin(), type_insensitive_op_list.end(),
                     [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
}

enum CastType { CAST_UP, CAST_DOWN, CAST_OTHER };
CastType GetCastType(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimCast)) {
    MS_LOG(EXCEPTION) << "Expect Cast node, but got " << common::AnfAlgo::GetCNodeName(node);
  }
  TypeId input_type = AnfAlgo::GetInputDeviceDataType(node, 0);
  TypeId output_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
  if (input_type == kNumberTypeFloat16 && output_type == kNumberTypeFloat32) {
    return CAST_UP;
  }
  if (input_type == kNumberTypeFloat32 && output_type == kNumberTypeFloat16) {
    return CAST_DOWN;
  }
  return CAST_OTHER;
}

std::vector<size_t> GetOpDataInputIndexes(const CNodePtr &node) {
  std::vector<size_t> op_input_indexes;
  if (node == nullptr || !IsTypeInsensitive(node)) {
    return op_input_indexes;
  }

  // Data input index starts from 0.
  if (IsPrimitiveCNode(node, prim::kPrimMaximum) || IsPrimitiveCNode(node, prim::kPrimMinimum)) {
    op_input_indexes = {0, 1};
  } else if (IsPrimitiveCNode(node, prim::kPrimSelect)) {
    op_input_indexes = {1, 2};
  } else {
    op_input_indexes = {0};
  }
  return op_input_indexes;
}

bool CheckInputTypeConsistent(const CNodePtr &node, const std::vector<size_t> &check_indexes, const TypeId &base_type) {
  MS_EXCEPTION_IF_NULL(node);

  // node's inputs at check_indexes should be of type base_type
  for (const auto &index : check_indexes) {
    if (AnfAlgo::GetInputDeviceDataType(node, index) != base_type) {
      return false;
    }
  }
  return true;
}

void SetNodeInfo(const CNodePtr &orig_node, const CNodePtr &new_node, const NodeIOInfo &node_io_info) {
  MS_EXCEPTION_IF_NULL(orig_node);
  MS_EXCEPTION_IF_NULL(new_node);

  auto node_name = common::AnfAlgo::GetCNodeName(new_node);
  auto orig_node_name = common::AnfAlgo::GetCNodeName(orig_node);
  if (orig_node_name != node_name) {
    MS_LOG(EXCEPTION) << "Can not process on different nodes " << orig_node_name << " and " << node_name;
  }

  AbstractBasePtr new_abstract{nullptr};
  if (node_io_info.outputs_type.empty()) {
    MS_LOG(EXCEPTION) << "Can not set empty output type of new node from " << orig_node->fullname_with_scope();
  }
  if (node_name == "Cast") {
    auto node_input = common::AnfAlgo::GetInputNode(new_node, 0);
    MS_EXCEPTION_IF_NULL(node_input);
    MS_EXCEPTION_IF_NULL(node_input->abstract());
    new_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(node_io_info.outputs_type[0]),
                                                              node_input->abstract()->BuildShape());
  } else {
    MS_EXCEPTION_IF_NULL(orig_node->abstract());
    new_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(node_io_info.outputs_type[0]),
                                                              orig_node->abstract()->BuildShape());
  }

  // Set abstract info
  new_node->set_abstract(new_abstract);
  // Set attrs
  common::AnfAlgo::CopyNodeAttrs(orig_node, new_node);
  // Set kernel build info
  new_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat(node_io_info.inputs_format);
  info_builder.SetInputsDeviceType(node_io_info.inputs_type);
  info_builder.SetOutputsFormat(node_io_info.outputs_format);
  info_builder.SetOutputsDeviceType(node_io_info.outputs_type);
  info_builder.SetKernelType(AnfAlgo::GetKernelType(orig_node));
  info_builder.SetOpPattern(AnfAlgo::GetOpPattern(orig_node));
  info_builder.SetFusionType(AnfAlgo::GetFusionType(orig_node));
  info_builder.SetProcessor(AnfAlgo::GetProcessor(orig_node));
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), new_node.get());
}
}  // namespace

void ReorderOps::SetTypeInsensitiveNodeInputs(const CNodePtr &node, const std::vector<size_t> &indexes,
                                              const std::vector<AnfNodePtr> &new_input_at_indexes,
                                              std::vector<AnfNodePtr> *new_inputs) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_inputs);
  if (indexes.size() != new_input_at_indexes.size()) {
    MS_LOG(EXCEPTION) << "indexes size " << indexes.size() << " is not equal to new_input_at_indexes size "
                      << new_input_at_indexes.size();
  }

  auto node_inputs_num = node->size();
  if (node_inputs_num == 0) {
    MS_LOG(EXCEPTION) << "Inputs num is 0 in node " << node->fullname_with_scope();
  }

  // node's inputs at indexes change to new_input_at_indexes
  if (!new_inputs->empty()) {
    new_inputs->resize(0);
  }
  new_inputs->push_back(node->input(0));
  mindspore::HashSet<size_t> indexes_set(indexes.begin(), indexes.end());
  size_t idx = 0;
  for (size_t i = 1; i < node_inputs_num; ++i) {
    size_t data_idx = i - 1;
    if (indexes_set.find(data_idx) == indexes_set.end()) {
      new_inputs->push_back(node->input(i));
    } else {
      new_inputs->push_back(new_input_at_indexes[idx++]);
    }
  }
}

void ReorderOps::SetTypeInsensitiveNodeInputsInfo(const CNodePtr &node, const std::vector<size_t> &indexes,
                                                  const std::vector<AnfNodePtr> &input_at_indexes,
                                                  NodeIOInfo *new_inputs_info, bool from_input) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_inputs_info);
  if (indexes.size() != input_at_indexes.size()) {
    MS_LOG(EXCEPTION) << "indexes size " << indexes.size() << " is not equal to new_input_at_indexes size "
                      << input_at_indexes.size();
  }

  auto node_inputs_num = node->size();
  if (node_inputs_num == 0) {
    MS_LOG(EXCEPTION) << "Inputs num is 0 in node " << node->fullname_with_scope();
  }

  // node's inputs info at indexes change to input_at_indexes's input or output info
  new_inputs_info->inputs_format.resize(0);
  new_inputs_info->inputs_type.resize(0);
  mindspore::HashSet<size_t> indexes_set(indexes.begin(), indexes.end());
  size_t idx = 0;
  for (size_t data_idx = 0; data_idx < node_inputs_num - 1; ++data_idx) {
    if (indexes_set.find(data_idx) == indexes_set.end()) {
      new_inputs_info->inputs_format.push_back(AnfAlgo::GetInputFormat(node, data_idx));
      new_inputs_info->inputs_type.push_back(AnfAlgo::GetInputDeviceDataType(node, data_idx));
    } else {
      if (from_input) {
        new_inputs_info->inputs_format.push_back(AnfAlgo::GetInputFormat(input_at_indexes[idx], 0));
        new_inputs_info->inputs_type.push_back(AnfAlgo::GetInputDeviceDataType(input_at_indexes[idx], 0));
      } else {
        new_inputs_info->inputs_format.push_back(AnfAlgo::GetOutputFormat(input_at_indexes[idx], 0));
        new_inputs_info->inputs_type.push_back(AnfAlgo::GetOutputDeviceDataType(input_at_indexes[idx], 0));
      }
      idx++;
    }
  }
}

bool ReorderOps::ReorderTypeInsensitiveCastDown(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                                                const CNodePtr &node) const {
  // Limitation:
  //   Current cast node is CAST_DOWN.
  //   Cast node will not change the input format.
  if (!IsPrimitiveCNode(node, prim::kPrimCast) || GetCastType(node) != CAST_DOWN ||
      AnfAlgo::GetInputFormat(node, 0) != AnfAlgo::GetOutputFormat(node, 0)) {
    return false;
  }

  auto large_type = AnfAlgo::GetInputDeviceDataType(node, 0);
  auto small_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
  auto pattern_output_format = AnfAlgo::GetOutputFormat(node, 0);

  auto node_input = common::AnfAlgo::GetInputNode(node, 0);
  auto type_insens_node = node_input->cast<CNodePtr>();
  // Limitation:
  //   Find type insensitive node before cast node.
  //   Type insensitive node is only used by current cast node.
  if (type_insens_node == nullptr || !IsTypeInsensitive(type_insens_node) ||
      mng->node_users()[type_insens_node].size() > 1) {
    return false;
  }

  auto op_input_indexes = GetOpDataInputIndexes(type_insens_node);
  // Limitation: Type insensitive node's inputs are the large type.
  if (op_input_indexes.empty() || !CheckInputTypeConsistent(type_insens_node, op_input_indexes, large_type)) {
    return false;
  }

  std::vector<AnfNodePtr> new_cast_nodes;
  for (const auto &index : op_input_indexes) {
    auto new_cast_node = func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())),
                                               common::AnfAlgo::GetInputNode(type_insens_node, index)});
    NodeIOInfo cast_io_info;
    cast_io_info.inputs_format.push_back(AnfAlgo::GetInputFormat(type_insens_node, index));
    cast_io_info.outputs_format = cast_io_info.inputs_format;
    cast_io_info.inputs_type.push_back(AnfAlgo::GetInputDeviceDataType(type_insens_node, index));
    cast_io_info.outputs_type.push_back(small_type);
    SetNodeInfo(node, new_cast_node, cast_io_info);
    new_cast_nodes.push_back(new_cast_node);
  }

  std::vector<AnfNodePtr> type_insens_node_new_inputs;
  SetTypeInsensitiveNodeInputs(type_insens_node, op_input_indexes, new_cast_nodes, &type_insens_node_new_inputs);
  NodeIOInfo type_insens_io_info;
  type_insens_io_info.outputs_format.push_back(pattern_output_format);
  type_insens_io_info.outputs_type.push_back(small_type);
  SetTypeInsensitiveNodeInputsInfo(type_insens_node, op_input_indexes, new_cast_nodes, &type_insens_io_info, false);
  auto new_type_insens_node = func_graph->NewCNode(type_insens_node_new_inputs);
  SetNodeInfo(type_insens_node, new_type_insens_node, type_insens_io_info);

  (void)mng->Replace(node, new_type_insens_node);
  return true;
}

bool ReorderOps::ReorderCastUpTypeInsensitive(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                                              const CNodePtr &node) const {
  if (!IsTypeInsensitive(node)) {
    return false;
  }

  // Limitation:
  //   Certain inputs of type insensitive node are cast node.
  //   Cast nodes are CAST_UP.
  //   Cast nodes will not change the input format.
  //   All these cast nodes are only used by current type insensitive node.
  std::vector<AnfNodePtr> cast_nodes;
  std::vector<AnfNodePtr> cast_input_nodes;
  auto op_input_indexes = GetOpDataInputIndexes(node);
  for (const auto &index : op_input_indexes) {
    auto node_input = common::AnfAlgo::GetInputNode(node, index);
    auto cast_node = node_input->cast<CNodePtr>();
    if (cast_node != nullptr && IsPrimitiveCNode(cast_node, prim::kPrimCast) && GetCastType(cast_node) == CAST_UP &&
        AnfAlgo::GetInputFormat(node, 0) == AnfAlgo::GetOutputFormat(node, 0) &&
        mng->node_users()[cast_node].size() == 1) {
      cast_nodes.push_back(cast_node);
      cast_input_nodes.push_back(common::AnfAlgo::GetInputNode(cast_node, 0));
    }
  }
  if (cast_nodes.empty() || cast_nodes.size() != op_input_indexes.size()) {
    return false;
  }

  auto small_type = AnfAlgo::GetInputDeviceDataType(cast_nodes[0], 0);
  auto large_type = AnfAlgo::GetOutputDeviceDataType(cast_nodes[0], 0);
  auto pattern_output_format = AnfAlgo::GetOutputFormat(node, 0);

  // Limitation: All these cast nodes cast same type to another type.
  if (!std::all_of(cast_nodes.begin(), cast_nodes.end(), [&small_type](const AnfNodePtr &cast_node) {
        return AnfAlgo::GetInputDeviceDataType(cast_node, 0) == small_type;
      })) {
    return false;
  }
  // Limitation: Type insensitive node's inputs have same data type.
  if (!CheckInputTypeConsistent(node, op_input_indexes, large_type)) {
    return false;
  }

  std::vector<AnfNodePtr> type_insens_node_new_inputs;
  SetTypeInsensitiveNodeInputs(node, op_input_indexes, cast_input_nodes, &type_insens_node_new_inputs);
  auto new_type_insens_node = func_graph->NewCNode(type_insens_node_new_inputs);
  NodeIOInfo type_insens_io_info;
  type_insens_io_info.outputs_format.push_back(pattern_output_format);
  type_insens_io_info.outputs_type.push_back(small_type);
  SetTypeInsensitiveNodeInputsInfo(node, op_input_indexes, cast_nodes, &type_insens_io_info, true);
  SetNodeInfo(node, new_type_insens_node, type_insens_io_info);

  auto new_cast_node =
    func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), new_type_insens_node});
  NodeIOInfo cast_io_info;
  cast_io_info.inputs_format.push_back(pattern_output_format);
  cast_io_info.outputs_format = cast_io_info.inputs_format;
  cast_io_info.inputs_type.push_back(small_type);
  cast_io_info.outputs_type.push_back(large_type);
  SetNodeInfo(cast_nodes[0]->cast<CNodePtr>(), new_cast_node, cast_io_info);

  (void)mng->Replace(node, new_cast_node);
  return true;
}

bool ReorderOps::ReorderCastTypeInsensitive(const FuncGraphPtr &func_graph) const {
  // Reorder cast node and type insensitive node in graph kernel sub-graph, this function has several limitations,
  //   see the comments that start will "Limitation:" in this file.
  // Limitation: Assuming the type insensitive node will not change the type of input nodes, otherwise it can be seen
  //   as another cast node in some sense, such as LessEqual operator, which performs on two inputs and output a
  //   a boolean result.
  auto mng = GkUtils::GetFuncGraphManager(func_graph);
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &anf_node : todos) {
    auto node = anf_node->cast<CNodePtr>();
    if (node == nullptr) {
      continue;
    }

    if (IsTypeInsensitive(node)) {
      // Reorder pattern 1: CastUp-TypeInsensitive --> TypeInsensitive-CastUp
      changed = ReorderCastUpTypeInsensitive(func_graph, mng, node) || changed;
    } else if (IsPrimitiveCNode(node, prim::kPrimCast)) {
      // Reorder pattern 2: TypeInsensitive-CastDown --> CastDown-TypeInsensitive
      changed = ReorderTypeInsensitiveCastDown(func_graph, mng, node) || changed;
    }
  }

  return changed;
}

bool ReorderOps::Run(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &anf_node : todos) {
    auto node = anf_node->cast<CNodePtr>();
    if (node == nullptr) {
      continue;
    }

    if (common::AnfAlgo::IsGraphKernel(node)) {
      auto sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      bool need_traverse = true;
      while (need_traverse) {
        need_traverse = ReorderCastTypeInsensitive(sub_func_graph);
        if (need_traverse) {
          changed = true;
        }
      }
    }
  }

  return changed;
}
}  // namespace mindspore::graphkernel
