/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/grad/dynamic_shape.h"

namespace mindspore {
namespace pynative {
namespace {
const size_t kMaxCacheDynamicShapeCellNum = 2;

bool IsValuePtrEqual(const ValuePtr &v1, const ValuePtr &v2) {
  if (v1 == v2) {
    return true;
  }
  if (v1 == nullptr || v2 == nullptr) {
    return false;
  }
  if (v1->isa<tensor::Tensor>() && v2->isa<tensor::Tensor>()) {
    return v1->cast<tensor::TensorPtr>()->ValueEqual(*(v2->cast<tensor::TensorPtr>()));
  }
  return *v1 == *v2;
}

bool IsDynamicDetectAbsChange(const AbstractBasePtr &old_abs, const AbstractBasePtr &new_abs) {
  if (old_abs == new_abs) {
    return false;
  }
  if (old_abs == nullptr || new_abs == nullptr) {
    MS_LOG(DEBUG) << "Graph is dynamic, old_abs is different with new_abs";
    return true;
  }
  if (!common::IsEqual(old_abs->BuildType(), new_abs->BuildType()) ||
      !common::IsEqual(old_abs->BuildShape(), new_abs->BuildShape())) {
    MS_LOG(DEBUG) << "Graph is dynamic, old_abs is different with new_abs, old abs: " << old_abs->ToString()
                  << " new abs: " << new_abs->ToString();
    return true;
  }
  return false;
}

bool IsDynamicDetectAbsChange(const abstract::AbstractBasePtrList &node_abs,
                              const abstract::AbstractBasePtrList &old_node_abs) {
  if (node_abs.size() != old_node_abs.size()) {
    MS_LOG(DEBUG) << "Graph is dynamic, node_abs size: " << node_abs.size()
                  << " old_node_abs size: " << old_node_abs.size();
    return true;
  }
  for (size_t i = 0; i < node_abs.size(); ++i) {
    if (IsDynamicDetectAbsChange(node_abs[i], old_node_abs[i])) {
      return true;
    }
  }
  return false;
}

bool IsDynamicDetectPrimChange(const PrimitivePtr &old_prim, const PrimitivePtr &new_prim) {
  if (old_prim == nullptr && new_prim == nullptr) {
    return false;
  }
  if (new_prim != nullptr && old_prim != nullptr) {
    return !common::IsEqual(old_prim, new_prim);
  }
  return true;
}

bool IsDynamicDetectNodeInfoChange(const NodeInfo &old_node_info, const NodeInfo &new_node_info) {
  if (new_node_info.grad_type == TensorGradType::kParameter &&
      (old_node_info.grad_type == TensorGradType::kParameter || old_node_info.grad_type == TensorGradType::kConstant)) {
    MS_EXCEPTION_IF_NULL(new_node_info.value);
    MS_EXCEPTION_IF_NULL(old_node_info.value);
    auto new_tensor = new_node_info.value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(new_tensor);
    auto old_tensor = old_node_info.value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(old_tensor);
    if (new_tensor->id() != old_tensor->id()) {
      MS_LOG(DEBUG) << "Graph is dynamic, new node info value: "
                    << (new_node_info.value != nullptr ? new_node_info.value->ToString() : "")
                    << " grad type: " << new_node_info.grad_type << " old node info value: "
                    << (old_node_info.value != nullptr ? old_node_info.value->ToString() : "")
                    << " grad type: " << old_node_info.grad_type;
      return true;
    }
    return false;
  }

  if (new_node_info.grad_type != old_node_info.grad_type) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node info grad type: " << new_node_info.grad_type
                  << " old node info grad type: " << old_node_info.grad_type;
    return true;
  }

  if (new_node_info.grad_type == TensorGradType::kOpOutput && new_node_info.op_index != old_node_info.op_index) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node info op_index: " << new_node_info.op_index
                  << " old node info op_index: " << old_node_info.op_index;
    return true;
  }

  if (new_node_info.grad_type == TensorGradType::kConstant &&
      !IsValuePtrEqual(new_node_info.value, old_node_info.value)) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node info value: "
                  << (new_node_info.value != nullptr ? new_node_info.value->ToString() : "")
                  << " grad type: " << new_node_info.grad_type
                  << " old node info value: " << (old_node_info.value != nullptr ? old_node_info.value->ToString() : "")
                  << " grad type: " << old_node_info.grad_type;
    return true;
  }

  return false;
}

void BuildDynamicDetectNodeInput(const ValuePtr &input, std::vector<std::pair<std::string, NodeInfo>> *node_inputs,
                                 const std::string &value_idx) {
  if (input->isa<tensor::Tensor>()) {
    NodeInfo node_info;
    auto tensor = input->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto auto_meta_data = tensor->auto_grad_meta_data();
    if (auto_meta_data == nullptr) {
      node_info.value = input;
      node_info.grad_type = TensorGradType::kConstant;
      (void)node_inputs->emplace_back(std::make_pair(value_idx, node_info));
      return;
    }
    node_info.grad_type = auto_meta_data->grad_type();
    node_info.op_index = auto_meta_data->op_index();
    if (node_info.grad_type == TensorGradType::kConstant || node_info.grad_type == TensorGradType::kParameter) {
      node_info.value = input;
    }
    (void)node_inputs->emplace_back(std::make_pair(value_idx, node_info));
  } else if (input->isa<ValueSequence>()) {
    auto value_sequence = input->cast<ValueSequencePtr>();
    for (size_t i = 0; i < value_sequence->value().size(); ++i) {
      const string &cur_idx = value_idx + std::to_string(i);
      BuildDynamicDetectNodeInput(value_sequence->value()[i], node_inputs, cur_idx);
    }
  } else if (input->isa<stub::StubNode>()) {
    auto stub_node = input->cast<stub::StubNodePtr>();
    MS_EXCEPTION_IF_NULL(stub_node);
    BuildDynamicDetectNodeInput(stub_node->WaitValue(), node_inputs, value_idx);
  } else {
    NodeInfo node_info;
    node_info.grad_type = TensorGradType::kConstant;
    node_info.value = input;
    (void)node_inputs->emplace_back(std::make_pair(value_idx, node_info));
  }
}

std::vector<std::pair<std::string, NodeInfo>> BuildDynamicDetectNodeInputs(const ValuePtrList &inputs) {
  std::vector<std::pair<std::string, NodeInfo>> node_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const string &tensor_idx = std::to_string(i);
    BuildDynamicDetectNodeInput(inputs[i], &node_inputs, tensor_idx);
  }
  return node_inputs;
}

bool IsDynamicDetectInputChange(const std::vector<std::pair<std::string, NodeInfo>> &old_inputs,
                                const std::vector<std::pair<std::string, NodeInfo>> &new_inputs) {
  if (old_inputs.size() != new_inputs.size()) {
    MS_LOG(DEBUG) << "Graph is dynamic, old_inputs size: " << old_inputs.size()
                  << "new_inputs size: " << new_inputs.size();
    return true;
  }
  for (size_t i = 0; i < old_inputs.size(); ++i) {
    std::string old_tensor_idx = old_inputs[i].first;
    auto old_node_info = old_inputs[i].second;
    std::string new_tensor_idx = new_inputs[i].first;
    auto new_node_info = new_inputs[i].second;
    if (old_tensor_idx != new_tensor_idx) {
      MS_LOG(DEBUG) << "Graph is dynamic, old_tensor_idx: " << old_tensor_idx << "new_tensor_idx: " << new_tensor_idx;
      return true;
    }
    if (IsDynamicDetectNodeInfoChange(old_node_info, new_node_info)) {
      MS_LOG(DEBUG) << "Graph is dynamic, old_node op index is: " << old_node_info.op_index
                    << " value is: " << (old_node_info.value != nullptr ? old_node_info.value->ToString() : "")
                    << " new_node op index is: " << new_node_info.op_index
                    << " value is: " << (new_node_info.value != nullptr ? new_node_info.value->ToString() : "");
      return true;
    }
  }
  return false;
}
}  // namespace

bool NodeDynamicDetect::CheckNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                         const DynamicDetectNodeInfoPtr &node) {
  std::unique_lock<std::mutex> lock(async_mutex_);
  MS_EXCEPTION_IF_NULL(top_cell);
  if (top_cell->use_dynamic_shape_process()) {
    top_cell->IncreaseOpIndex();
    return true;
  }

  const size_t node_idx = top_cell->op_index();
  bool use_dynamic_shape_process = IsNodeDynamic(top_cell, inputs, node, node_idx);
  top_cell->IncreaseOpIndex();
  if (use_dynamic_shape_process) {
    MS_LOG(INFO) << "Set use_dynamic_shape_process: " << use_dynamic_shape_process;
    top_cell->set_use_dynamic_shape_process(use_dynamic_shape_process);
    py::gil_scoped_acquire gil_acquire;
    (void)cell_id_with_dynamic_detect_nodes_.erase(top_cell->obj_id_with_grad_order());
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE)) {
      MS_LOG(WARNING) << "Detect dynamic shape or dynamic graph structure, the python stack is:";
      py::gil_scoped_acquire acquire_gil;
      py::exec(R"(
                  import traceback
                  traceback.print_stack()
                  )");
    }
  }
  return use_dynamic_shape_process;
}

bool NodeDynamicDetect::IsNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                      const DynamicDetectNodeInfoPtr &node, size_t node_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (top_cell->is_need_save_dynamic_detect_nodes()) {
    SaveDynamicDetectNodeInfoInFirstTime(top_cell, inputs, node, node_idx);
    // The net is regarded as a static net by default in the first time.
    return false;
  }

  MS_LOG(DEBUG) << "Check node " << (node->op_prim != nullptr ? node->op_prim->name() : "") << " node_idx: " << node_idx
                << ", is_jit_node: " << node->is_graph_node << ", graph_phase:" << node->graph_phase
                << " obj_id_with_grad_order:" << top_cell->obj_id_with_grad_order()
                << ", cell id:" << top_cell->cell_id();
  const auto &dynamic_nodes =
    cell_id_with_dynamic_detect_nodes_[top_cell->obj_id_with_grad_order()][top_cell->cell_id()];
  if (node_idx >= dynamic_nodes.size()) {
    MS_LOG(DEBUG) << "Old dynamic_nodes size: " << dynamic_nodes.size() << ", cur node_idx is: " << node_idx
                  << ", graph is dynamic.";
    return true;
  }

  // 1.Detect jit phase
  const DynamicDetectNodeInfoPtr &old_node_info = dynamic_nodes[node_idx];
  if (node->is_graph_node) {
    if (!old_node_info->is_graph_node || node->graph_phase != old_node_info->graph_phase) {
      MS_LOG(DEBUG) << "Graph is dynamic, old is_graph_node: " << old_node_info->is_graph_node
                    << " new is_graph_node: " << node->is_graph_node << " old graph_phase "
                    << old_node_info->graph_phase << " new graph_phase: " << node->graph_phase;
      return true;
    }
    return false;
  }

  if (IsDynamicDetectPrimChange(old_node_info->op_prim, node->op_prim)) {
    MS_LOG(DEBUG) << "Graph is dynamic, old node prim: " << old_node_info->op_prim->name()
                  << " new node prim: " << (node->op_prim != nullptr ? node->op_prim->name() : "")
                  << " node_idx: " << node_idx;
    return true;
  }

  // Compare input abs
  if (IsDynamicDetectAbsChange(old_node_info->input_abs, node->input_abs)) {
    return true;
  }

  // Compare out abs
  if (IsDynamicDetectAbsChange(old_node_info->out_abs, node->out_abs)) {
    return true;
  }

  // Get input
  node->inputs = BuildDynamicDetectNodeInputs(inputs);

  // Compare input
  if (IsDynamicDetectInputChange(old_node_info->inputs, node->inputs)) {
    return true;
  }
  return false;
}

void NodeDynamicDetect::SaveDynamicDetectNodeInfoInFirstTime(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                                             const DynamicDetectNodeInfoPtr &node, size_t node_idx) {
  MS_EXCEPTION_IF_NULL(node);
  node->inputs = BuildDynamicDetectNodeInputs(inputs);
  (void)cell_id_with_dynamic_detect_nodes_[top_cell->obj_id_with_grad_order()][top_cell->cell_id()].emplace_back(node);
  MS_LOG(DEBUG) << "Save node " << (node->op_prim != nullptr ? node->op_prim->name() : "")
                << " firstly, node_idx: " << node_idx << ", is_jit_node: " << node->is_graph_node
                << ", graph_phase:" << node->graph_phase
                << " obj_id_with_grad_order:" << top_cell->obj_id_with_grad_order()
                << ", cell id:" << top_cell->cell_id();
}

bool NodeDynamicDetect::IsNeedSaveDynamicDetectNodes(const TopCellInfoPtr &top_cell, bool use_dynamic_shape_process) {
  if (use_dynamic_shape_process) {
    // top cell is already dynamic shape, no need save nodes.
    return false;
  }
  MS_EXCEPTION_IF_NULL(top_cell);
  auto cell_iter = cell_id_with_dynamic_detect_nodes_.find(top_cell->obj_id_with_grad_order());
  if (cell_iter == cell_id_with_dynamic_detect_nodes_.end()) {
    // Cell is not found in cell_id_with_dynamic_detect_nodes_, need save nodes first.
    return true;
  }

  const auto &cell_infos = cell_iter->second;
  if (cell_infos.size() == 1) {
    // top_cell->cell_id() is cell id with inputs shape, if cell id in cell_id_with_dynamic_detect_nodes_
    // id same with top_cell->cell_id(), no need save nodes.
    return cell_infos.begin()->first != top_cell->cell_id();
  } else if (cell_infos.size() == kMaxCacheDynamicShapeCellNum) {
    auto cell_infos_iter = cell_infos.find(top_cell->cell_id());
    if (cell_infos_iter == cell_infos.end()) {
      // cell_id_with_dynamic_detect_nodes_ has two cell id already, current cell is is different
      // with them. So set_use_dynamic_shape_process for top cell.
      top_cell->set_use_dynamic_shape_process(true);
      (void)cell_id_with_dynamic_detect_nodes_.erase(top_cell->obj_id_with_grad_order());
      MS_LOG(INFO) << "Set use_dynamic_shape_process: " << use_dynamic_shape_process << ", already cached "
                   << cell_infos.size() << " top cell, cur top cell shape is different:" << top_cell->cell_id();
    }
  } else {
    MS_LOG(EXCEPTION) << "cell_info.size():" << cell_infos.size() << " is invalid";
  }
  return false;
}
}  // namespace pynative
}  // namespace mindspore
