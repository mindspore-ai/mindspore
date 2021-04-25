/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/graph_kernel/reorder_ops.h"
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_set>
#include "base/core_ops.h"
#include "utils/utils.h"
#include "utils/log_adapter.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {
bool IsTypeInsensitive(const CNodePtr &node) {
  // Nodes that will change the input data type will not seen as type insensitive nodes.
  static std::unordered_set<PrimitivePtr> type_insensitive_op_list{
    prim::KPrimTransData, prim::kPrimTranspose, prim::kPrimExpandDims, prim::kPrimReshape,
    prim::kPrimSqueeze,   prim::kPrimTile,      prim::kPrimNeg,        prim::kPrimRelu,
    prim::kPrimMaximum,   prim::kPrimMinimum,   prim::kPrimSelect};

  return std::any_of(type_insensitive_op_list.begin(), type_insensitive_op_list.end(),
                     [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
}

enum CastType { CAST_UP, CAST_DOWN, CAST_OTHER };
CastType GetCastType(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimCast)) {
    MS_LOG(EXCEPTION) << "Only process for Cast!";
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

void SetNodeInfo(const CNodePtr &orig_node, const CNodePtr &new_node, const TypeId &node_type) {
  MS_EXCEPTION_IF_NULL(orig_node);
  MS_EXCEPTION_IF_NULL(new_node);

  auto node_name = AnfAlgo::GetCNodeName(new_node);
  auto orig_node_name = AnfAlgo::GetCNodeName(orig_node);
  if (orig_node_name != node_name) {
    MS_LOG(EXCEPTION) << "Can not process on different nodes " << orig_node_name << " and " << node_name;
  }

  AbstractBasePtr new_abstract{nullptr};
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type{node_type};
  KernelType kernel_type{AnfAlgo::GetKernelType(orig_node)};
  kernel::OpPattern op_pattern{AnfAlgo::GetOpPattern(orig_node)};
  kernel::FusionType fusion_type{AnfAlgo::GetFusionType(orig_node)};
  kernel::Processor processor{AnfAlgo::GetProcessor(orig_node)};

  auto node_data_inputs_num = AnfAlgo::GetInputNum(new_node);
  for (size_t i = 0; i < node_data_inputs_num; ++i) {
    auto node_input = AnfAlgo::GetInputNode(new_node, i);
    auto node_input_format = AnfAlgo::GetOutputFormat(node_input, 0);
    auto node_input_type = AnfAlgo::GetOutputDeviceDataType(node_input, 0);
    inputs_format.push_back(node_input_format);
    inputs_device_type.push_back(node_input_type);
  }
  if (node_name == "Cast") {
    auto node_input = AnfAlgo::GetInputNode(new_node, 0);
    new_abstract =
      std::make_shared<abstract::AbstractTensor>(TypeIdToType(node_type), node_input->abstract()->BuildShape());
    outputs_format.push_back(AnfAlgo::GetOutputFormat(node_input, 0));
  } else {
    new_abstract =
      std::make_shared<abstract::AbstractTensor>(TypeIdToType(node_type), orig_node->abstract()->BuildShape());
    outputs_format.push_back(AnfAlgo::GetOutputFormat(orig_node, 0));
  }

  // Set abstract info
  new_node->set_abstract(new_abstract);
  // Set attrs
  AnfAlgo::CopyNodeAttrs(orig_node, new_node);
  // Set kernel build info
  new_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat(inputs_format);
  info_builder.SetInputsDeviceType(inputs_device_type);
  info_builder.SetOutputsFormat(outputs_format);
  info_builder.SetOutputsDeviceType(outputs_device_type);
  info_builder.SetKernelType(kernel_type);
  info_builder.SetOpPattern(op_pattern);
  info_builder.SetFusionType(fusion_type);
  info_builder.SetProcessor(processor);
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), new_node.get());
}
}  // namespace

void ReorderOps::SetTypeInsensitiveNodeInputs(const CNodePtr &node, const std::vector<size_t> &indexes,
                                              const std::vector<AnfNodePtr> &new_input_at_indexes,
                                              std::vector<AnfNodePtr> *new_inputs) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_inputs);
  if (indexes.size() != new_input_at_indexes.size()) {
    MS_LOG(EXCEPTION) << "indexes size " << indexes.size() << " is not equal to new_input_at_indexes size "
                      << new_input_at_indexes.size();
  }
  if (!new_inputs->empty()) {
    new_inputs->resize(0);
  }

  // node's inputs at indexes change to new_input_at_indexes
  std::unordered_set<size_t> indexes_set(indexes.begin(), indexes.end());
  auto node_inputs_num = node->size();
  size_t idx = 0;
  for (size_t i = 0; i < node_inputs_num; ++i) {
    if (indexes_set.find(i) == indexes_set.end()) {
      new_inputs->push_back(node->input(i));
    } else {
      new_inputs->push_back(new_input_at_indexes[idx++]);
    }
  }
}

bool ReorderOps::ReorderTypeInsensitiveCastDown(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                                                const CNodePtr &node) {
  // Limitation: Current cast node is CAST_DOWN.
  if (!IsPrimitiveCNode(node, prim::kPrimCast) || GetCastType(node) != CAST_DOWN) {
    return false;
  }

  auto node_input = AnfAlgo::GetInputNode(node, 0);
  auto type_insens_node = node_input->cast<CNodePtr>();
  // Limitation:
  //   Find type insensitive node before cast node.
  //   Type insensitive node is only used by current cast node.
  if (type_insens_node == nullptr || !IsTypeInsensitive(type_insens_node) ||
      mng->node_users()[type_insens_node].size() > 1) {
    return false;
  }

  auto cast_input_type = AnfAlgo::GetInputDeviceDataType(node, 0);
  auto cast_out_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
  auto op_input_indexes = GetOpDataInputIndexes(type_insens_node);
  // Limitation: Type insensitive node's inputs have same data type.
  if (op_input_indexes.empty() || !CheckInputTypeConsistent(type_insens_node, op_input_indexes, cast_input_type)) {
    return false;
  }

  std::vector<AnfNodePtr> new_cast_nodes;
  for (const auto &index : op_input_indexes) {
    auto new_cast_node =
      func_graph->NewCNode({NewValueNode(prim::kPrimCast), AnfAlgo::GetInputNode(type_insens_node, index)});
    SetNodeInfo(node, new_cast_node, cast_out_type);
    new_cast_nodes.push_back(new_cast_node);
  }

  std::transform(op_input_indexes.begin(), op_input_indexes.end(), op_input_indexes.begin(),
                 [](const size_t &idx) { return idx + 1; });

  std::vector<AnfNodePtr> type_insens_node_new_inputs;
  SetTypeInsensitiveNodeInputs(type_insens_node, op_input_indexes, new_cast_nodes, &type_insens_node_new_inputs);
  auto new_type_insens_node = func_graph->NewCNode(type_insens_node_new_inputs);
  SetNodeInfo(type_insens_node, new_type_insens_node, cast_out_type);

  (void)mng->Replace(node, new_type_insens_node);
  return true;
}

bool ReorderOps::ReorderCastUpTypeInsensitive(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                                              const CNodePtr &node) {
  if (!IsTypeInsensitive(node)) {
    return false;
  }

  // Limitation:
  //   Certain inputs of type insensitive node are cast node.
  //   Cast nodes are CAST_UP.
  //   All these cast nodes are only used by current type insensitive node.
  std::vector<CNodePtr> cast_nodes;
  std::vector<AnfNodePtr> cast_input_nodes;
  auto op_input_indexes = GetOpDataInputIndexes(node);
  for (const auto &index : op_input_indexes) {
    auto node_input = AnfAlgo::GetInputNode(node, index);
    auto cast_node = node_input->cast<CNodePtr>();
    if (cast_node != nullptr && IsPrimitiveCNode(cast_node, prim::kPrimCast) && GetCastType(cast_node) == CAST_UP &&
        mng->node_users()[cast_node].size() == 1) {
      cast_nodes.push_back(cast_node);
      cast_input_nodes.push_back(AnfAlgo::GetInputNode(cast_node, 0));
    }
  }
  if (cast_nodes.empty() || cast_nodes.size() != op_input_indexes.size()) {
    return false;
  }

  auto cast_input_type = AnfAlgo::GetInputDeviceDataType(cast_nodes[0], 0);
  auto cast_out_type = AnfAlgo::GetOutputDeviceDataType(cast_nodes[0], 0);
  // Limitation: All these cast nodes cast same type to another type.
  if (!std::all_of(cast_nodes.begin(), cast_nodes.end(), [&cast_input_type](const CNodePtr &cast_node) {
        return AnfAlgo::GetInputDeviceDataType(cast_node, 0) == cast_input_type;
      })) {
    return false;
  }
  // Limitation: Type insensitive node's inputs have same data type.
  if (!CheckInputTypeConsistent(node, op_input_indexes, cast_out_type)) {
    return false;
  }

  std::transform(op_input_indexes.begin(), op_input_indexes.end(), op_input_indexes.begin(),
                 [](const size_t &idx) { return idx + 1; });

  std::vector<AnfNodePtr> type_insens_node_new_inputs;
  SetTypeInsensitiveNodeInputs(node, op_input_indexes, cast_input_nodes, &type_insens_node_new_inputs);
  auto new_type_insens_node = func_graph->NewCNode(type_insens_node_new_inputs);
  SetNodeInfo(node, new_type_insens_node, cast_input_type);

  auto new_cast_node = func_graph->NewCNode({NewValueNode(prim::kPrimCast), new_type_insens_node});
  SetNodeInfo(cast_nodes[0], new_cast_node, cast_out_type);

  (void)mng->Replace(node, new_cast_node);
  return true;
}

bool ReorderOps::ReorderCastTypeInsensitive(const FuncGraphPtr &func_graph) {
  // Reorder cast node and type insensitive node in graph kernel sub-graph, this function has several limitations,
  //   see the comments that start will "Limitation:" in this file.
  // Limitation: Assuming the type insensitive node will not change the type of input nodes, otherwise it can be seen
  //   as another cast node in some sense, such as LessEqual operator, which performs on two inputs and output a
  //   a boolean result.
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
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
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }

  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &anf_node : todos) {
    auto node = anf_node->cast<CNodePtr>();
    if (node == nullptr) {
      continue;
    }

    if (AnfAlgo::IsGraphKernel(node)) {
      auto sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
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
}  // namespace opt
}  // namespace mindspore
