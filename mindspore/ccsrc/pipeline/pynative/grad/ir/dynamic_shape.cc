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

#include "pipeline/pynative/grad/ir/dynamic_shape.h"
#include <algorithm>
#include "pipeline/pynative/pynative_utils.h"

namespace mindspore {
namespace pynative {
namespace {
constexpr auto kIsFeatureMapOutput = "IsFeatureMapOutput";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
constexpr size_t kMaxCacheDynamicShapeCellNum = 2;

bool IsValuePtrEqual(const ValuePtr &v1, const ValuePtr &v2) {
  if (v1 == v2) {
    return true;
  }
  if (v1 == nullptr || v2 == nullptr) {
    return false;
  }
  if (v1->isa<tensor::BaseTensor>() && v2->isa<tensor::BaseTensor>()) {
    return v1->cast<tensor::BaseTensorPtr>()->ValueEqual(*(v2->cast<tensor::BaseTensorPtr>()));
  }
  return *v1 == *v2;
}

bool IsDynamicDetectPrimChange(const PrimitivePtr &old_prim, const PrimitivePtr &new_prim) {
  if (old_prim == nullptr && new_prim == nullptr) {
    return false;
  }
  // Use kernel graph will add kIsFeatureMapOutput and kIsFeatureMapOutput attr.
  // After run first step, prim may be changed. So, prim check exclude them.
  if (old_prim != nullptr && old_prim->HasAttr(kIsFeatureMapOutput)) {
    old_prim->EraseAttr(kIsFeatureMapOutput);
    old_prim->EraseAttr(kIsFeatureMapInputList);
  }
  if (new_prim != nullptr && old_prim != nullptr) {
    return !common::IsEqual(old_prim, new_prim);
  }
  return true;
}

bool IsNodeInfoChange(const NodeInfo &old_node_info, const NodeInfo &new_node_info) {
  size_t input_size = old_node_info.seq_node.size();
  if (input_size != new_node_info.seq_node.size()) {
    MS_LOG(DEBUG) << "Graph is dynamic, input is tuple, but old seq node size " << input_size << ", new seq node size "
                  << new_node_info.seq_node.size();
    return true;
  }
  for (size_t i = 0; i < input_size; ++i) {
    if (IsNodeInfoChange(old_node_info.seq_node[i], new_node_info.seq_node[i])) {
      return true;
    }
  }

  if (new_node_info.grad_type != old_node_info.grad_type) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node grad type: " << new_node_info.grad_type
                  << ", old node grad type: " << old_node_info.grad_type;
    return true;
  }

  if (new_node_info.op_index != old_node_info.op_index) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node use op_index: " << new_node_info.op_index
                  << ", old node use op_index: " << old_node_info.op_index;
    return true;
  }

  if (new_node_info.grad_type == InputType::kParameter && old_node_info.grad_type == InputType::kParameter) {
    MS_EXCEPTION_IF_NULL(new_node_info.value);
    MS_EXCEPTION_IF_NULL(old_node_info.value);
    auto new_tensor = new_node_info.value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(new_tensor);
    auto old_tensor = old_node_info.value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(old_tensor);
    if (new_tensor->id() != old_tensor->id()) {
      MS_LOG(DEBUG) << "Graph is dynamic, new parameter name: "
                    << (new_tensor->param_info() != nullptr
                          ? new_tensor->param_info()->name() + ", requires_grad " +
                              std::to_string(new_tensor->param_info()->requires_grad())
                          : "")
                    << ", new parameter id: " << new_tensor->id() << ", old node parameter name: "
                    << (old_tensor->param_info() != nullptr
                          ? old_tensor->param_info()->name() + ", requires_grad " +
                              std::to_string(old_tensor->param_info()->requires_grad())
                          : "")
                    << ", old parameter id: " << old_tensor->id();
      return true;
    }
    return false;
  }

  if (new_node_info.grad_type == InputType::kConstant && !IsValuePtrEqual(new_node_info.value, old_node_info.value)) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node constant value: "
                  << (new_node_info.value != nullptr ? new_node_info.value->ToString() : "")
                  << ", old node value: " << (old_node_info.value != nullptr ? old_node_info.value->ToString() : "");
    return true;
  }
  return false;
}

bool IsInputsNodeInfoChange(const std::vector<NodeInfo> &old_inputs_node_info,
                            const std::vector<NodeInfo> &new_inputs_node_info) {
  size_t input_size = old_inputs_node_info.size();
  if (input_size != new_inputs_node_info.size()) {
    MS_LOG(DEBUG) << "Graph is dynamic, old_inputs size: " << input_size
                  << "new_inputs size: " << new_inputs_node_info.size();
    return true;
  }
  for (size_t i = 0; i < input_size; ++i) {
    if (IsNodeInfoChange(old_inputs_node_info[i], new_inputs_node_info[i])) {
      return true;
    }
  }
  return false;
}

NodeInfo GetNodeInfoFromValue(const ValuePtr &input) {
  if (input->isa<tensor::BaseTensor>()) {
    NodeInfo node_info;
    auto tensor = input->cast<tensor::BaseTensorPtr>();
    auto auto_meta_data = tensor->auto_grad_meta_data();
    // Scalar tensor
    if (auto_meta_data == nullptr) {
      node_info.grad_type = InputType::kConstant;
      node_info.value = input;
      return node_info;
    }

    // Tensor
    node_info.grad_type = auto_meta_data->input_type();
    node_info.op_index = auto_meta_data->op_index();
    if (node_info.grad_type == InputType::kConstant || node_info.grad_type == InputType::kParameter) {
      node_info.value = input;
    }
    return node_info;
  }
  if (input->isa<ValueSequence>()) {
    NodeInfo node_info;
    const auto &value_sequence = input->cast<ValueSequencePtr>();
    for (const auto &i : value_sequence->value()) {
      (void)node_info.seq_node.emplace_back(GetNodeInfoFromValue(i));
    }
    return node_info;
  }
  if (input->isa<stub::StubNode>()) {
    auto stub_node = input->cast<stub::StubNodePtr>();
    MS_EXCEPTION_IF_NULL(stub_node);
    return GetNodeInfoFromValue(stub_node->WaitValue());
  }
  NodeInfo node_info;
  node_info.grad_type = InputType::kConstant;
  node_info.value = input;
  return node_info;
}

struct CompareBasedOnAbstract {
  static bool IsNodeChange(const ValuePtrList &inputs, const DynamicDetectNodeInfoPtr &old_node,
                           const DynamicDetectNodeInfoPtr &new_node) {
    // Compare input abs
    if (IsDynamicDetectAbsChange(old_node->abs_compare_info.input_abs, new_node->abs_compare_info.input_abs)) {
      return true;
    }

    // Compare out abs
    if (IsDynamicDetectAbsChange(old_node->abs_compare_info.out_abs, new_node->abs_compare_info.out_abs)) {
      return true;
    }

    // Get input
    BuildDynamicDetectInputsNodeInfo(new_node, inputs);

    // Compare input
    return IsInputsNodeInfoChange(old_node->abs_compare_info.inputs, new_node->abs_compare_info.inputs);
  }

  static bool IsDynamicDetectAbsChange(const AbstractBasePtr &old_abs, const AbstractBasePtr &new_abs) {
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
                    << ", new abs: " << new_abs->ToString();
      return true;
    }
    return false;
  }

  static bool IsDynamicDetectAbsChange(const abstract::AbstractBasePtrList &node_abs,
                                       const abstract::AbstractBasePtrList &old_node_abs) {
    if (node_abs.size() != old_node_abs.size()) {
      MS_LOG(DEBUG) << "Graph is dynamic, node_abs size: " << node_abs.size()
                    << ", old_node_abs size: " << old_node_abs.size();
      return true;
    }
    for (size_t i = 0; i < node_abs.size(); ++i) {
      if (IsDynamicDetectAbsChange(node_abs[i], old_node_abs[i])) {
        return true;
      }
    }
    return false;
  }

  static void BuildDynamicDetectInputsNodeInfo(const DynamicDetectNodeInfoPtr &node, const ValuePtrList &inputs) {
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(node->abs_compare_info.inputs),
                   [](const auto &item) { return GetNodeInfoFromValue(item); });
  }
};

struct CompareBasedOnValueSimpleInfo {
  static bool IsNodeChange(const ValuePtrList &inputs, const DynamicDetectNodeInfoPtr &old_node,
                           const DynamicDetectNodeInfoPtr &new_node) {
    BuildInputsValueSimpleInfo(new_node, inputs);
    return IsInputsChange(old_node->value_compare_info, new_node->value_compare_info);
  }

  static void BuildInputsValueSimpleInfo(const DynamicDetectNodeInfoPtr &node, const ValuePtrList &inputs) {
    size_t input_size = inputs.size();
    node->value_compare_info.input_value_simple_info.size_ = input_size;
    node->value_compare_info.input_value_simple_info.shape_vector_.reserve(input_size);
    node->value_compare_info.input_value_simple_info.dtype_vector_.reserve(input_size);
    node->value_compare_info.input_value_simple_info.object_type_vector_.reserve(input_size);
    for (const auto &input : inputs) {
      node->value_compare_info.inputs.emplace_back(GetNodeInfoFromValue(input));

      (void)node->value_compare_info.input_value_simple_info.shape_vector_.emplace_back(
        PyNativeAlgo::Common::GetShapeFromValue(input));
      auto [dtype, obj_type] = PyNativeAlgo::Common::GetTypeFromValue(input);
      (void)node->value_compare_info.input_value_simple_info.dtype_vector_.emplace_back(dtype);
      (void)node->value_compare_info.input_value_simple_info.object_type_vector_.emplace_back(obj_type);
    }
  }

  static bool IsInputsChange(const ValueCompareInfo &old_value_compare_info,
                             const ValueCompareInfo &new_value_compare_info) {
    if (IsInputsNodeInfoChange(old_value_compare_info.inputs, new_value_compare_info.inputs)) {
      return true;
    }
    return IsValueSimpleInfoChange(old_value_compare_info.input_value_simple_info,
                                   new_value_compare_info.input_value_simple_info);
  }

  template <typename T1, typename T2>
  static bool IsNotEuqal(const T1 &old_input, const T2 &new_input) {
    return old_input != new_input;
  }

  template <typename T1, typename T2>
  static bool IsNotEuqal(const std::shared_ptr<T1> &old_input, const std::shared_ptr<T2> &new_input) {
    MS_EXCEPTION_IF_NULL(old_input);
    MS_EXCEPTION_IF_NULL(new_input);
    return old_input->type_id() != new_input->type_id();
  }

  static bool IsValueSimpleInfoChange(const ValueSimpleInfo &old_input_simple_info,
                                      const ValueSimpleInfo &new_input_simple_info) {
    if (old_input_simple_info.size_ != new_input_simple_info.size_) {
      MS_LOG(DEBUG) << "Graph is dynamic, old_input_simple_info size: " << old_input_simple_info.size_
                    << ", new_input_simple_info size: " << new_input_simple_info.size_;
      return true;
    }
    for (size_t i = 0; i < old_input_simple_info.size_; ++i) {
      if (IsNotEuqal(old_input_simple_info.shape_vector_[i], new_input_simple_info.shape_vector_[i]) ||
          IsNotEuqal(old_input_simple_info.dtype_vector_[i], new_input_simple_info.dtype_vector_[i]) ||
          IsNotEuqal(old_input_simple_info.object_type_vector_[i], new_input_simple_info.object_type_vector_[i])) {
        MS_LOG(DEBUG) << "Graph is dynamic, old input simple info: " << ValueSimpleInfoToString(old_input_simple_info)
                      << ", new input simple info: " << ValueSimpleInfoToString(new_input_simple_info);
        return true;
      }
    }
    return false;
  }
};

void UpdateAbsCache(const std::string &arg_id, const ValuePtr &v, const abstract::BaseShapePtr &base_shape,
                    const abstract::AbstractBasePtr &abs, size_t index) {
  auto update_abs = abs;
  if (update_abs == nullptr) {
    MS_EXCEPTION_IF_NULL(v);
    auto input_tensor = v->cast<tensor::BaseTensorPtr>();
    // Just tensor work in unknown shape
    if (input_tensor == nullptr) {
      return;
    }
    MS_EXCEPTION_IF_NULL(base_shape);
    update_abs = std::make_shared<abstract::AbstractTensor>(input_tensor->Dtype(), base_shape);
  }
  MS_LOG(DEBUG) << "Set arg " << index << ", id " << arg_id << ", to dynamic abs: " << update_abs->ToString();
  const auto &infer = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->infer_operation();
  infer->UpdateNodeAbsCacheById(arg_id, update_abs);
}

bool GetUnknownShape(const ShapeVector &cur_shape, const ShapeVector &pre_top_cell_shape, ShapeVector *new_shape) {
  // Dynamic rank
  if (cur_shape.size() != pre_top_cell_shape.size()) {
    MS_LOG(INFO) << "Cur shape size " << cur_shape.size() << " is not equal to top cell arg shape size "
                 << pre_top_cell_shape.size();
    (void)new_shape->emplace_back(abstract::Shape::kShapeRankAny);
    return true;
  }
  // Dynamic shape
  for (size_t j = 0; j < cur_shape.size(); ++j) {
    if (cur_shape[j] == pre_top_cell_shape[j]) {
      (void)new_shape->emplace_back(cur_shape[j]);
    } else {
      (void)new_shape->emplace_back(abstract::Shape::kShapeDimAny);
    }
  }
  // All shape can not be actual, which indicates static shape.
  if (!IsDynamicShape(*new_shape)) {
    MS_LOG(DEBUG) << "All shape are actual, is static shape. Cur shape " << cur_shape << ", elem shape "
                  << pre_top_cell_shape << ", and new shape is " << new_shape;
    return false;
  }
  return true;
}

bool IsMatch(const ShapeVector &cur_shape, const ShapeVector &pre_top_cell_shape) {
  if (cur_shape.size() != pre_top_cell_shape.size() && !pre_top_cell_shape.empty() &&
      pre_top_cell_shape[kIndex0] != abstract::Shape::kShapeRankAny) {
    MS_LOG(DEBUG) << "Cur shape size " << cur_shape.size() << " is not equal to pre top cell arg shape size "
                  << pre_top_cell_shape.size();
    return false;
  }
  // Dynamic rank or dynamic shape
  for (size_t i = 0; i < cur_shape.size(); ++i) {
    if (cur_shape[i] != pre_top_cell_shape[i] && pre_top_cell_shape[i] != abstract::Shape::kShapeDimAny) {
      MS_LOG(DEBUG) << "Cur shape " << cur_shape[i] << " can not match pre top cell shape " << pre_top_cell_shape[i];
      return false;
    }
  }
  return true;
}
}  // namespace

py::object DynamicShape::GetDynamicInput(const py::object &actual_input) {
  if (py::isinstance<py::tuple>(actual_input)) {
    auto tuple_actual_args = py::cast<py::tuple>(actual_input);
    size_t args_size = tuple_actual_args.size();
    py::tuple dyn_shape_args = py::tuple(args_size);
    for (size_t i = 0; i < args_size; ++i) {
      dyn_shape_args[i] = GetDynamicInput(tuple_actual_args[i]);
    }
    return dyn_shape_args;
  }
  if (py::isinstance<py::list>(actual_input)) {
    auto list_actual_args = py::cast<py::list>(actual_input);
    size_t args_size = list_actual_args.size();
    py::list dyn_shape_args;
    for (size_t i = 0; i < args_size; ++i) {
      dyn_shape_args.append(GetDynamicInput(list_actual_args[i]));
    }
    return dyn_shape_args;
  }
  if (py::isinstance<tensor::BaseTensor>(actual_input)) {
    const auto &infer = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->infer_operation();
    auto tensor_ptr = py::cast<tensor::BaseTensorPtr>(actual_input);
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    auto dyn_compile_tensor = std::make_shared<tensor::BaseTensor>(tensor_ptr->data_type(), tensor_ptr->shape_c());
    const auto &abs = infer->GetNodeAbsById(PyNativeAlgo::PyParser::GetIdByPyObj(actual_input));
    if (abs != nullptr) {
      auto base_shape = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(base_shape);
      if (base_shape->IsDynamic()) {
        dyn_compile_tensor->set_base_shape(base_shape);
      }
    }
    return PyNativeAlgo::DataConvert::ValueToPyObj(dyn_compile_tensor);
  }
  return actual_input;
}

void DynamicShape::SaveUnknownShapeAbsFromJit(const ValuePtr &v, const AbstractBasePtr &abs, size_t index) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(abs);
  if (v->isa<ValueSequence>() && abs->isa<abstract::AbstractSequence>()) {
    const auto &v_seq = v->cast<ValueSequencePtr>();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    if (v_seq->size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Obj tuple size " << v_seq->size() << ", but abstract tuple size " << abs_seq->size();
    }
    for (size_t i = 0; i < v_seq->size(); ++i) {
      SaveUnknownShapeAbsFromJit(v_seq->value()[i], abs_seq->elements()[i], index);
    }
  } else if (v->isa<tensor::BaseTensor>() && abs->isa<abstract::AbstractTensor>()) {
    if (abs->BuildShape()->IsDynamic()) {
      UpdateAbsCache(PyNativeAlgo::Common::GetIdByValue(v), v, nullptr, abs, ++index);
    }
  } else {
    MS_LOG(EXCEPTION) << "Not match: obj " << v->ToString() << " and abs " << abs->ToString();
  }
}

void NodeDynamicDetect::CheckNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                         const DynamicDetectNodeInfoPtr &node) {
  bool node_is_dynamic = false;
  bool has_bprop_cut_op = top_cell->has_bprop_cut_op();
  if (!has_bprop_cut_op) {
    node_is_dynamic = IsNodeDynamic(top_cell, inputs, node);
  } else {
    MS_LOG(INFO) << "Set use_dynamic_shape_process by bprop cut op";
  }
  if (has_bprop_cut_op || node_is_dynamic) {
    MS_LOG(INFO) << "Set use_dynamic_shape_process: " << true;
    top_cell->set_use_dynamic_shape_process(true);
    py::gil_scoped_acquire gil_acquire;
    (void)cell_id_with_dynamic_detect_nodes_.erase(top_cell->obj_id_with_grad_order());
  }
  if (node_is_dynamic) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE)) {
      MS_LOG(WARNING) << "Detect dynamic shape or dynamic graph structure, the python stack is: ";
      py::gil_scoped_acquire acquire_gil;
      py::exec(R"(
                  import traceback
                  traceback.print_stack()
                  )");
    }
  }
}

bool NodeDynamicDetect::IsNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                      const DynamicDetectNodeInfoPtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (top_cell->is_need_save_dynamic_detect_nodes()) {
    SaveDynamicDetectNodeInfoInFirstTime(top_cell, inputs, node);
    // The net is regarded as a static net by default in the first time.
    return false;
  }

  MS_LOG(DEBUG) << "Check node " << (node->op_prim != nullptr ? node->op_prim->name() : "")
                << " node_idx: " << node->op_index << ", graph_phase: " << node->graph_phase
                << ", obj_id_with_grad_order: " << top_cell->obj_id_with_grad_order()
                << ", cell id: " << top_cell->cell_id();

  const auto &dynamic_nodes =
    cell_id_with_dynamic_detect_nodes_[top_cell->obj_id_with_grad_order()][top_cell->cell_id()];
  // op index must begin from 0
  if (node->op_index >= dynamic_nodes.size()) {
    MS_LOG(DEBUG) << "Old dynamic_nodes size: " << dynamic_nodes.size() << ", cur node_idx is: " << node->op_index
                  << ", graph is dynamic.";
    return true;
  }

  const DynamicDetectNodeInfoPtr &old_node_info = dynamic_nodes[node->op_index];
  // 1.Detect jit phase
  if (node->graph_phase != old_node_info->graph_phase) {
    MS_LOG(DEBUG) << "Graph is dynamic, old graph_phase " << old_node_info->graph_phase
                  << ", new graph_phase: " << node->graph_phase;
    return true;
  }

  // 2.Detect prim
  if (IsDynamicDetectPrimChange(old_node_info->op_prim, node->op_prim)) {
    MS_LOG(DEBUG) << "Graph is dynamic, old node prim: "
                  << (old_node_info->op_prim != nullptr
                        ? old_node_info->op_prim->name() + ", attr: " + old_node_info->op_prim->GetAttrsText()
                        : "")
                  << ", new node prim: "
                  << (node->op_prim != nullptr ? node->op_prim->name() + ", attr: " + node->op_prim->GetAttrsText()
                                               : "")
                  << ", node_idx: " << node->op_index;
    return true;
  }

  // 3.Detect inputs
  if (node->is_value_compare) {
    return CompareBasedOnValueSimpleInfo::IsNodeChange(inputs, old_node_info, node);
  }
  return CompareBasedOnAbstract::IsNodeChange(inputs, old_node_info, node);
}

void NodeDynamicDetect::SaveDynamicDetectNodeInfoInFirstTime(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                                             const DynamicDetectNodeInfoPtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->is_value_compare) {
    CompareBasedOnValueSimpleInfo::BuildInputsValueSimpleInfo(node, inputs);
  } else {
    CompareBasedOnAbstract::BuildDynamicDetectInputsNodeInfo(node, inputs);
  }
  (void)cell_id_with_dynamic_detect_nodes_[top_cell->obj_id_with_grad_order()][top_cell->cell_id()].emplace_back(node);
  MS_LOG(DEBUG) << "Save node " << (node->op_prim != nullptr ? node->op_prim->name() : "")
                << " firstly, node_idx: " << node->op_index << ", graph_phase: " << node->graph_phase
                << ", obj_id_with_grad_order: " << top_cell->obj_id_with_grad_order()
                << ", cell id: " << top_cell->cell_id();
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
  }
  if (cell_infos.size() == kMaxCacheDynamicShapeCellNum) {
    auto cell_infos_iter = cell_infos.find(top_cell->cell_id());
    if (cell_infos_iter == cell_infos.end()) {
      // cell_id_with_dynamic_detect_nodes_ has two cell id already, current cell is is different
      // with them. So set_use_dynamic_shape_process for top cell.
      top_cell->set_use_dynamic_shape_process(true);
      (void)cell_id_with_dynamic_detect_nodes_.erase(top_cell->obj_id_with_grad_order());
      MS_LOG(INFO) << "Set use_dynamic_shape_process: " << use_dynamic_shape_process << ", already cached "
                   << cell_infos.size() << " top cell, cur top cell shape is different: " << top_cell->cell_id();
    }
  } else {
    MS_LOG(EXCEPTION) << "cell_info.size(): " << cell_infos.size() << " is invalid";
  }
  return false;
}

void TopCellUnknownShapeDetect::SetDynamicInput(const py::object &obj, const py::args &args) {
  const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
  // After first step, set inputs no need work again. Because the top cell of first step is already unknown shape and
  // follow step will keep unknown shape always, special input_signature
  if (obj_with_by_inputs_.find(obj_id) != obj_with_by_inputs_.end()) {
    MS_LOG(DEBUG) << "Obj " << obj_id << " has done set inputs before";
    return;
  }
  auto &arg_base_shape_vec = obj_id_args_info_by_set_inputs_[obj_id];
  size_t args_size = args.size();
  arg_base_shape_vec.reserve(args_size);
  for (size_t i = 0; i < args_size; ++i) {
    (void)arg_base_shape_vec.emplace_back(PyNativeAlgo::DataConvert::PyObjToValue(args[i])->ToAbstract()->BuildShape());
  }
  TryChangeTopCellToUnknownShape(obj_id, arg_base_shape_vec, false);
  (void)obj_with_by_inputs_.emplace(obj_id);
}

void TopCellUnknownShapeDetect::TryChangeTopCellToUnknownShape(const std::string &obj_id,
                                                               const abstract::BaseShapePtrList &arg_base_shape_vec,
                                                               bool is_auto_detect) {
  const auto &grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  if (is_auto_detect) {
    // From auto detect
    auto &top_cell_list = grad_executor->already_run_top_cell();
    const auto it = std::find_if(top_cell_list.begin(), top_cell_list.end(), [&obj_id](const auto &elem) {
      return elem.second->input_args_info() != nullptr && elem.second->input_args_info()->obj_id == obj_id;
    });
    if (it != top_cell_list.end()) {
      // Pre top cell is already unknown shape, check current top cell can match it
      if (it->second->is_unknown_shape() && CanFindMatchedUnknownShapeTopCell(it->second, arg_base_shape_vec)) {
        MS_LOG(DEBUG) << "Pre top cell has already been unknown shape and can match current top cell";
        ChangeTopCellToUnknownShape(grad_executor->top_cell(), it->second->input_args_info()->input_arg_base_shape_vec);
        return;
      }
      // If not match before, compare shape and change current top cell do unknown shape
      if (SetTopCellUnknownShape(grad_executor->top_cell(), it->second, arg_base_shape_vec)) {
        (void)top_cell_list.erase(it);
        return;
      }
    } else {
      // Set inputs, first step top cell working here
      const auto item = obj_id_args_info_by_set_inputs_.find(grad_executor->top_cell()->input_args_info()->obj_id);
      if (item != obj_id_args_info_by_set_inputs_.end()) {
        const auto &input_args_info = grad_executor->top_cell()->input_args_info();
        UpdateUnknownShapeAbsCache(input_args_info->input_arg_id_vec, input_args_info->input_arg_value_vec,
                                   item->second);
        (void)obj_id_args_info_by_set_inputs_.erase(item);
        return;
      }
      // C1.set_inputs, run C1(x); C2 is top cell, and run C2(x).
      if (std::any_of(arg_base_shape_vec.begin(), arg_base_shape_vec.end(),
                      [](const abstract::BaseShapePtr &base_shape) { return base_shape->IsDynamic(); })) {
        MS_LOG(DEBUG) << "Top cell is unknown shape now";
        grad_executor->top_cell()->set_is_unknown_shape(true);
      }
    }
  } else {
    // From set inputs. Has not create top cell yet
    if (grad_executor->TopCellHasNotBeenCreate()) {
      return;
    }
    // Jit, top cell create first, then set inputs run
    const auto item = obj_id_args_info_by_set_inputs_.find(grad_executor->top_cell()->input_args_info()->obj_id);
    if (item != obj_id_args_info_by_set_inputs_.end()) {
      MS_LOG(DEBUG) << "Get jit set inputs";
      ChangeTopCellToUnknownShape(grad_executor->top_cell(), arg_base_shape_vec);
      (void)obj_id_args_info_by_set_inputs_.erase(item);
    }
  }
}

void TopCellUnknownShapeDetect::UpdateUnknownShapeAbsCache(const std::vector<string> &input_arg_id_vec,
                                                           const std::vector<ValuePtr> &input_arg_value_vec,
                                                           const std::vector<abstract::BaseShapePtr> &args_base_shape) {
  for (size_t i = 0; i < args_base_shape.size(); i++) {
    MS_EXCEPTION_IF_NULL(args_base_shape[i]);
    MS_EXCEPTION_IF_NULL(input_arg_value_vec[i]);
    if (args_base_shape[i]->IsDynamic()) {
      if (args_base_shape[i]->isa<abstract::Shape>()) {
        UpdateAbsCache(input_arg_id_vec[i], input_arg_value_vec[i], args_base_shape[i], nullptr, i);
      } else if (args_base_shape[i]->isa<abstract::SequenceShape>()) {
        // Input arg is list or tuple
        const auto &seq_shape = args_base_shape[i]->cast<abstract::SequenceShapePtr>();
        const auto &seq_v = input_arg_value_vec[i]->cast<ValueSequencePtr>();
        MS_EXCEPTION_IF_NULL(seq_v);
        if (seq_v->size() != seq_shape->size()) {
          MS_LOG(EXCEPTION) << "Sequence value size " << seq_v->size() << " is not equal to seq shape size "
                            << seq_shape->size();
        }
        std::vector<std::string> id_vec;
        PyNativeAlgo::Common::SplitString(input_arg_id_vec[i], &id_vec);
        if (id_vec.size() != seq_shape->size()) {
          MS_LOG(EXCEPTION) << "Id size " << id_vec.size() << " is not equal to seq shape size " << seq_shape->size();
        }
        for (size_t j = 0; j < seq_shape->size(); ++j) {
          UpdateAbsCache(id_vec[j], seq_v->value()[j], seq_shape->shape()[j], nullptr, i + j);
        }
      }
    }
  }
}

void TopCellUnknownShapeDetect::UpdateArgsAbsToUnknownShapeAbs(const py::object &obj, const py::args &args) {
  if (obj_id_args_info_by_set_inputs_.empty()) {
    return;
  }

  const auto &grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  bool top_cell_has_not_been_create = grad_executor->TopCellHasNotBeenCreate();
  // Top cell is already unknown shape
  if (!top_cell_has_not_been_create && grad_executor->top_cell()->is_unknown_shape()) {
    return;
  }

  // Current cell is has no set_inputs
  const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
  const auto it = obj_id_args_info_by_set_inputs_.find(obj_id);
  if (it == obj_id_args_info_by_set_inputs_.end()) {
    return;
  }

  // Common cell args id and value not create in ParsePyArgsToInputArgsInfo, need get them now.
  // Update current cell id cache which maybe used for top cell
  const auto &args_id_v = PyNativeAlgo::PyParser::GetArgsIdAndValue(args);
  UpdateUnknownShapeAbsCache(args_id_v.first, args_id_v.second, it->second);

  // C1.set_inputs, run C1(x); C2 is top cell, and run C2(x).
  if (top_cell_has_not_been_create) {
    // Has not create top cell yet
    (void)obj_id_args_info_by_set_inputs_.erase(it);
    return;
  }

  // C1 is top cell, run C1(x); C2 set_inputs, and run C2(x).
  UpdatePossibleTopCellToUnknownShape(grad_executor->top_cell(), args_id_v.first, it->second);
  (void)obj_id_args_info_by_set_inputs_.erase(it);
}

void TopCellUnknownShapeDetect::UpdatePossibleTopCellToUnknownShape(const TopCellInfoPtr &cur_top_cell,
                                                                    const std::vector<string> &cur_arg_id_vec,
                                                                    const abstract::BaseShapePtrList &cur_args_shape) {
  MS_LOG(DEBUG) << "Update possible top cell";
  auto cur_top_cell_base_shape_vec = cur_top_cell->input_args_info()->input_arg_base_shape_vec;
  const auto &cur_top_cell_id_vec = cur_top_cell->input_args_info()->input_arg_id_vec;
  bool need_change_top_cell_info = false;
  // Check top cell args id is the same with current set inputs cell. If dynamic shape, update top cell to unknown shape
  for (size_t i = 0; i < cur_arg_id_vec.size(); ++i) {
    auto it = std::find(cur_top_cell_id_vec.begin(), cur_top_cell_id_vec.end(), cur_arg_id_vec[i]);
    if (it != cur_top_cell_id_vec.end() && cur_args_shape[i]->IsDynamic()) {
      auto id_index = it - cur_top_cell_id_vec.begin();
      cur_top_cell_base_shape_vec[id_index] = cur_args_shape[i];
      need_change_top_cell_info = true;
    }
  }
  // Change current top cell info
  if (need_change_top_cell_info) {
    cur_top_cell->ChangeTopCellInfo(cur_top_cell_base_shape_vec);
  }
}

bool TopCellUnknownShapeDetect::CanFindMatchedUnknownShapeTopCell(const TopCellInfoPtr &pre_top_cell,
                                                                  const abstract::BaseShapePtrList &cur_args_shape) {
  for (size_t i = 0; i < cur_args_shape.size(); ++i) {
    const auto &cur_shape = cur_args_shape[i];
    const auto &pre_top_cell_shape = pre_top_cell->input_args_info()->input_arg_base_shape_vec[i];
    MS_EXCEPTION_IF_NULL(cur_shape);
    MS_EXCEPTION_IF_NULL(pre_top_cell_shape);
    if (cur_shape->isa<abstract::Shape>() && pre_top_cell_shape->isa<abstract::Shape>()) {
      if (!IsMatch(cur_shape->cast<abstract::ShapePtr>()->shape(),
                   pre_top_cell_shape->cast<abstract::ShapePtr>()->shape())) {
        return false;
      }
    } else if (cur_shape->isa<abstract::SequenceShape>() && pre_top_cell_shape->isa<abstract::SequenceShape>()) {
      // Input arg is list or tuple
      const auto &cur_shape_seq = cur_shape->cast<abstract::SequenceShapePtr>();
      const auto &top_cell_shape_seq = pre_top_cell_shape->cast<abstract::SequenceShapePtr>();
      size_t cur_shape_size = cur_shape_seq->size();
      if (cur_shape_size != top_cell_shape_seq->size()) {
        MS_LOG(DEBUG) << "The " << i << "th args shape size is not the same, cur is " << cur_shape_seq->size()
                      << " and the elem is " << top_cell_shape_seq->size();
        return false;
      }
      for (size_t j = 0; j < cur_shape_size; ++j) {
        MS_EXCEPTION_IF_NULL(cur_shape_seq->shape()[j]);
        MS_EXCEPTION_IF_NULL(top_cell_shape_seq->shape()[j]);
        if (!IsMatch(cur_shape_seq->shape()[j]->cast<abstract::ShapePtr>()->shape(),
                     top_cell_shape_seq->shape()[j]->cast<abstract::ShapePtr>()->shape())) {
          return false;
        }
      }
    }
  }
  return true;
}

void TopCellUnknownShapeDetect::ChangeTopCellToUnknownShape(const TopCellInfoPtr &top_cell,
                                                            const abstract::BaseShapePtrList &args_unknown_shape) {
  if (top_cell->input_args_info()->input_arg_base_shape_vec.size() != args_unknown_shape.size()) {
    MS_LOG(EXCEPTION) << "Top cell args base shape size "
                      << top_cell->input_args_info()->input_arg_base_shape_vec.size()
                      << " is not equal to update unknown shape size " << args_unknown_shape.size();
  }
  UpdateUnknownShapeAbsCache(top_cell->input_args_info()->input_arg_id_vec,
                             top_cell->input_args_info()->input_arg_value_vec, args_unknown_shape);
  top_cell->ChangeTopCellInfo(args_unknown_shape);
}

bool TopCellUnknownShapeDetect::SetTopCellUnknownShape(const TopCellInfoPtr &cur_top_cell,
                                                       const TopCellInfoPtr &pre_top_cell,
                                                       const abstract::BaseShapePtrList &args_shape) {
  abstract::BaseShapePtrList args_unknown_shape;
  args_unknown_shape.reserve(args_shape.size());
  for (size_t i = 0; i < args_shape.size(); ++i) {
    const auto &cur_shape = args_shape[i];
    const auto &pre_top_cell_shape = pre_top_cell->input_args_info()->input_arg_base_shape_vec[i];
    MS_EXCEPTION_IF_NULL(cur_shape);
    MS_EXCEPTION_IF_NULL(pre_top_cell_shape);
    if (cur_shape->isa<abstract::Shape>() && pre_top_cell_shape->isa<abstract::Shape>()) {
      ShapeVector new_shape;
      auto has_unknown = GetUnknownShape(cur_shape->cast<abstract::ShapePtr>()->shape(),
                                         pre_top_cell_shape->cast<abstract::ShapePtr>()->shape(), &new_shape);
      if (has_unknown) {
        (void)args_unknown_shape.emplace_back(std::make_shared<abstract::Shape>(new_shape));
      }
    } else if (cur_shape->isa<abstract::SequenceShape>() && pre_top_cell_shape->isa<abstract::SequenceShape>()) {
      // Input arg is list or tuple
      const auto &cur_shape_seq = cur_shape->cast<abstract::SequenceShapePtr>();
      MS_EXCEPTION_IF_NULL(cur_shape_seq);
      const auto &pre_top_cell_shape_seq = pre_top_cell_shape->cast<abstract::SequenceShapePtr>();
      size_t cur_shape_size = cur_shape_seq->size();
      if (cur_shape_size != pre_top_cell_shape_seq->size()) {
        MS_LOG(DEBUG) << "The " << i << "th args shape size is not the same, cur is " << cur_shape_seq->size()
                      << " and the elem is " << pre_top_cell_shape_seq->size();
      }
      abstract::BaseShapePtrList shape_ptr_list;
      for (size_t j = 0; j < cur_shape_size; ++j) {
        const auto &cur_shape_elem = cur_shape_seq->shape()[j]->cast<abstract::ShapePtr>();
        const auto &pre_top_cell_shape_elem = pre_top_cell_shape_seq->shape()[j]->cast<abstract::ShapePtr>();
        MS_EXCEPTION_IF_NULL(pre_top_cell_shape_elem);
        ShapeVector new_shape;
        auto has_unknown = GetUnknownShape(cur_shape_elem->shape(), pre_top_cell_shape_elem->shape(), &new_shape);
        if (has_unknown) {
          (void)shape_ptr_list.emplace_back(std::make_shared<abstract::Shape>(new_shape));
        }
      }
      if (shape_ptr_list.size() == cur_shape_size) {
        (void)args_unknown_shape.emplace_back(std::make_shared<abstract::TupleShape>(shape_ptr_list));
      }
    } else {
      MS_LOG(DEBUG) << "The " << i << "th args shape type is not the same, cur is " << cur_shape->ToString()
                    << " and the elem is " << pre_top_cell_shape->ToString();
      return false;
    }
  }
  if (args_unknown_shape.size() == args_shape.size()) {
    ChangeTopCellToUnknownShape(cur_top_cell, args_unknown_shape);
    return true;
  }
  return false;
}
}  // namespace pynative
}  // namespace mindspore
