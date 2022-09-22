/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/pynative_utils.h"
#include "ir/tensor.h"
#include "runtime/device/device_address.h"

namespace mindspore {
namespace pynative {
void TopCellInfo::SetCellSelfInfoForTopCell(const py::object &cell, const py::args &args) {
  std::vector<std::string> args_id;
  std::vector<abstract::ShapePtr> args_shape;
  std::vector<TypePtr> args_type;
  for (size_t i = 0; i < args.size(); ++i) {
    auto value = PyNativeAlgo::DataConvert::PyObjToValue(args[i]);
    MS_EXCEPTION_IF_NULL(value);
    auto abs = value->ToAbstract();
    auto shape_ptr = abs->BuildShape()->cast<abstract::ShapePtr>();
    if (shape_ptr == nullptr) {
      return;
    }
    (void)args_id.emplace_back(PyNativeAlgo::PyParser::GetIdByPyObj(args[i]));
    (void)args_shape.emplace_back(shape_ptr);
    (void)args_type.emplace_back(abs->BuildType());
  }
  set_cell_self_info(
    std::make_shared<CellSelfInfo>(PyNativeAlgo::PyParser::GetIdByPyObj(cell), args_id, args_shape, args_type));
}

bool TopCellInfo::IsSubCell(const std::string &cell_id) const {
  if (sub_cell_list_.empty()) {
    MS_LOG(DEBUG) << "The sub cell list is empty, there is no sub cell";
    return false;
  }
  return sub_cell_list_.find(cell_id) != sub_cell_list_.end();
}

void TopCellInfo::RecordCellBackwardHookOp(const std::string &cell_order, const AnfNodePtr &hook_op) {
  MS_EXCEPTION_IF_NULL(hook_op);
  (void)cell_backward_hook_op_[cell_order].emplace_back(hook_op);
  constexpr size_t cell_backward_hook_max_num = 2;
  if (cell_backward_hook_op_[cell_order].size() > cell_backward_hook_max_num) {
    MS_LOG(EXCEPTION) << "Cell order: " << cell_order << " only has two backward hook op.";
  }
}

void TopCellInfo::CheckSubCellHookChanged() {
  if (!hook_changed_) {
    for (const auto &sub_cell : sub_cell_list_) {
      const auto sub_cell_id = sub_cell.substr(0, sub_cell.find('_'));
      if (sub_cell_hook_changed_.find(sub_cell_id) != sub_cell_hook_changed_.end()) {
        hook_changed_ = true;
        break;
      }
    }
  }
  sub_cell_hook_changed_.clear();
}

void TopCellInfo::ClearDeviceMemory() const {
  MS_LOG(DEBUG) << "Clear device memory in value nodes of bprop graph, top cell: " << cell_id_;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == kCPUDevice) {
    MS_LOG(DEBUG) << "No need to clear device address when run in CPU device.";
    return;
  }
  // Get all tensors obj in value node of running graph
  std::vector<tensor::TensorPtr> tensors_in_bprop_graph;
  MS_EXCEPTION_IF_NULL(resource_);
  const auto &bprop_graph = resource_->func_graph();
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &value_node_list = bprop_graph->value_nodes();
  for (const auto &elem : value_node_list) {
    auto &node = elem.first;
    MS_EXCEPTION_IF_NULL(node);
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    TensorValueToTensor(value_node->value(), &tensors_in_bprop_graph);
  }
  for (const auto &tensor : tensors_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "Clear device address for tensor: " << tensor->ToString();
    auto device_sync = tensor->device_address();
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
    if (device_address == nullptr) {
      continue;
    }
    if (!device_address->from_persistent_mem()) {
      tensor->set_device_address(nullptr);
    }
  }
}

void TopCellInfo::Clear() {
  MS_LOG(DEBUG) << "Clear top cell info. Cell id " << cell_id_;
  op_num_ = 0;
  is_dynamic_structure_ = false;
  vm_compiled_ = false;
  ms_function_flag_ = false;
  is_init_kpynative_ = false;
  need_compile_graph_ = false;
  forward_already_run_ = false;
  input_args_id_.clear();
  all_op_info_.clear();
  resource_ = nullptr;
  df_builder_ = nullptr;
  fg_ = nullptr;
  k_pynative_cell_ptr_ = nullptr;
  graph_info_map_.clear();
  sub_cell_list_.clear();
  op_info_with_tensor_id_.clear();
  tensor_id_with_tensor_object_.clear();
  op_info_with_ms_func_forward_tensors_.clear();
}

void TopCellInfo::ResetTopCellInfo(const py::args &args) {
  op_num_ = 0;
  all_op_info_.clear();
  set_forward_already_run(true);
  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) + "_";
  }
  set_input_args_id(input_args_id);
}

void TopCellInfo::SaveOpInfo(const std::string &op_info, const std::vector<tensor::TensorPtr> &op_out_tensors) {
  const auto &op_info_with_tensor_id = this->op_info_with_tensor_id();
  if (op_info_with_tensor_id.find(op_info) != op_info_with_tensor_id.end()) {
    MS_LOG(EXCEPTION) << "Top cell: " << cell_id_ << " records op info with tensor id, but get op info " << op_info
                      << " in op_info_with_tensor_id map";
  }
  // Record the relationship between the forward op and its output tensor id
  (void)std::for_each(op_out_tensors.begin(), op_out_tensors.end(), [this, &op_info](const tensor::TensorPtr &tensor) {
    this->SetOpInfoWithTensorId(op_info, tensor->id());
  });
}

void TopCellInfo::RecordGradOpInfo(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::string input_args_info;
  auto is_param_require_grad = [](const BaseOpRunInfo &op_run_info, const size_t &idx) -> bool {
    if (op_run_info.input_mask[idx] == kParameterDataTensorMask || idx >= op_run_info.input_tensor.size()) {
      return false;
    }

    if (!op_run_info.input_tensor[idx]->is_parameter() || op_run_info.input_tensor[idx]->param_info() == nullptr) {
      // Maybe mask is kValueNodeTensorMask.
      return true;
    }

    return op_run_info.input_tensor[idx]->param_info()->requires_grad();
  };

  // Record input args info (weight or data)
  for (size_t i = 0; i < op_run_info->base_op_run_info.input_mask.size(); i++) {
    if (is_param_require_grad(op_run_info->base_op_run_info, i)) {
      input_args_info += "w";
      continue;
    }
    input_args_info += "d";
  }

  // Record op name and index
  op_run_info->op_info.clear();
  op_run_info->op_info += op_run_info->base_op_run_info.op_name + "-" + std::to_string(op_num_) + "-" + input_args_info;
  // The out shape(not dynamic shape) is added to determine those ops that change the shape
  bool is_dynamic_shape_out = !dynamic_shape() || op_run_info->base_op_run_info.has_dynamic_output;
  const auto &out_abs = op_run_info->base_op_run_info.abstract;
  if (is_dynamic_shape_out && out_abs != nullptr) {
    auto shape = out_abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (!shape->isa<abstract::NoShape>() && !shape->IsDimZero()) {
      op_run_info->op_info += "-" + shape->ToString();
    }
  }
  AppendAllOpInfo(op_run_info->op_info);
  IncreaseOpNum();
}

void TopCellInfo::SetParamNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id,
                                                const ParameterPtr &param) const {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->params[id] = param;
}
void TopCellInfo::SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                           int64_t index) const {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->node_map[id] = std::make_pair(node, std::vector<int64_t>{index});
}
void TopCellInfo::SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                           const std::vector<int64_t> &index) const {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->node_map[id] = std::make_pair(node, index);
}

void TopCellInfo::SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const ValuePtr &v, const AnfNodePtr &node,
                                             bool is_param) {
  MS_EXCEPTION_IF_NULL(v);
  if (!v->isa<ValueSequence>()) {
    return;
  }
  auto tuple = v->cast<ValueSequencePtr>()->value();
  auto tuple_size = static_cast<int64_t>(tuple.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    // tuple slice used size_t
    auto id = PyNativeAlgo::Common::GetIdByValue(tuple[static_cast<size_t>(i)]);
    if (is_param && node->isa<Parameter>()) {
      auto param = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      SetParamNodeMapInGraphInfoMap(g, id, param);
    }
    SetNodeMapInGraphInfoMap(g, id, node, i);
    SetTupleItemArgsToGraphInfoMap(g, tuple[i], node, std::vector<int64_t>{i}, is_param);
  }
}

void TopCellInfo::SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const ValuePtr &v, const AnfNodePtr &node,
                                                 const std::vector<int64_t> &index_sequence, bool is_param) {
  MS_EXCEPTION_IF_NULL(v);
  if (!v->isa<ValueSequence>()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto tuple = v->cast<ValueSequencePtr>()->value();
  auto tuple_size = static_cast<int64_t>(tuple.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    std::vector<int64_t> tmp = index_sequence;
    (void)tmp.emplace_back(i);
    // tuple slice used size_t
    auto id = PyNativeAlgo::Common::GetIdByValue(tuple[static_cast<size_t>(i)]);
    if (is_param && node->isa<Parameter>()) {
      auto param = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      SetParamNodeMapInGraphInfoMap(g, id, param);
    }
    SetNodeMapInGraphInfoMap(g, id, node, tmp);
    SetTupleItemArgsToGraphInfoMap(g, tuple[i], node, tmp, is_param);
  }
}

std::string TopCellInfo::GetAlreadyRunCellId(const std::string &cell_id) const {
  std::string already_run_cell_id(cell_id);
  size_t grad_order = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->grad_order();
  already_run_cell_id += std::to_string(grad_order == 0 ? 1 : grad_order);
  already_run_cell_id += "_" + grad_operation_;
  MS_LOG(DEBUG) << "Get already run top cell id " << already_run_cell_id;
  return already_run_cell_id;
}

void TopCellInfo::ChangeTopCellInfo(size_t args_size) {
  std::string new_cell_id = this->cell_self_info()->cell_self_id;
  for (size_t i = 0; i < args_size; ++i) {
    new_cell_id += "_" + this->cell_self_info()->args_shape[i]->ToString();
    new_cell_id += this->cell_self_info()->args_type[i]->ToString();
  }
  MS_LOG(DEBUG) << "Change top cell " << this->cell_id() << " to be dynamic " << new_cell_id;
  set_cell_id(new_cell_id);
  set_already_run_cell_id(GetAlreadyRunCellId(new_cell_id));
}
}  // namespace pynative
}  // namespace mindspore
