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
#include "include/backend/device_address.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pynative {
void TopCellInfo::RecordCellBackwardHookOp(const std::string &cell_id, const AnfNodePtr &hook_op) {
  MS_EXCEPTION_IF_NULL(hook_op);
  MS_LOG(DEBUG) << "Get cell register backward hook, id " << cell_id;
  (void)cell_backward_hook_op_[cell_id].emplace_back(hook_op);
}

void TopCellInfo::GetOpInfo(const FrontendOpRunInfoPtr &op_run_info, bool is_jit_graph) const {
  // Dynamic shape no need do value node replace
  if (use_dynamic_shape_process() && !is_jit_graph) {
    return;
  }
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->op_info.clear();
  op_run_info->op_info += op_run_info->base_op_run_info.op_name + "-" + std::to_string(op_index_);
}

void TopCellInfo::UpdateTopCellInfo(bool forward_already_run, bool need_compile_graph, bool vm_compile) {
  need_compile_graph_ = need_compile_graph;
  forward_already_run_ = forward_already_run;
  vm_compile_ = vm_compile;
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
  // Top cell has already call Clear(), this maybe happen in no need compile grad scenario
  if (resource_ == nullptr) {
    MS_LOG(DEBUG) << "This top cell " << this << " has already been clear";
    return;
  }

  const auto &bprop_graph = resource_->func_graph();
  if (bprop_graph == nullptr) {
    return;
  }
  const auto &value_node_list = bprop_graph->value_nodes();
  // Get all tensors obj in value node of running graph
  std::vector<tensor::BaseTensorPtr> tensors_in_bprop_graph;
  for (const auto &elem : value_node_list) {
    auto &node = elem.first;
    MS_EXCEPTION_IF_NULL(node);
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    TensorValueToTensor(value_node->value(), &tensors_in_bprop_graph);
  }
  for (const auto &tensor : tensors_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto device_sync = tensor->device_address();
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
    if (device_address == nullptr) {
      continue;
    }
    if (!device_address->from_persistent_mem() && !tensor->is_parameter() && !IsOutputTensor(tensor)) {
      // Parameters can not be cleaned up. In the case of Parameter(Tensor(xxx).view(xxx), requires_grad=False),
      // the param will be converted to value node into bprop graph. Tensor will be zero after cleaning.
      MS_LOG(DEBUG) << "Clear device address for tensor: " << tensor->id() << ", device address " << device_address
                    << ", device ptr " << device_address->GetPtr();
      tensor->set_device_address(nullptr);
    }
  }
}

void TopCellInfo::AddMetaGradInfo(const tensor::BaseTensorPtr &tensor, const AutoGradMetaDataPtr &auto_grad_meta_data) {
  meta_grad_info_[tensor] = auto_grad_meta_data;
}

void TopCellInfo::BackUpValueMetaGradInfo(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor_value = value->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = tensor_value->auto_grad_meta_data();
    if (auto_grad_meta_data != nullptr) {
      meta_grad_info_[tensor_value] = auto_grad_meta_data;
    }
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    for (const auto &elem : value_seq->value()) {
      BackUpValueMetaGradInfo(elem);
    }
  } else if (value->isa<stub::StubNode>()) {
    auto stub_node = value->cast<stub::StubNodePtr>();
    MS_EXCEPTION_IF_NULL(stub_node);
    BackUpValueMetaGradInfo(stub_node->WaitValue());
  }
}

void TopCellInfo::ClearValueMetaGradInfo(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor_value = value->cast<tensor::BaseTensorPtr>();
    // Hook register before op run
    if (tensor_value->auto_grad_meta_data() != nullptr && tensor_value->auto_grad_meta_data()->is_register_hook()) {
      return;
    }
    tensor_value->set_auto_grad_meta_data(nullptr);
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    for (const auto &elem : value_seq->value()) {
      ClearValueMetaGradInfo(elem);
    }
  } else if (value->isa<stub::StubNode>()) {
    auto stub_node = value->cast<stub::StubNodePtr>();
    MS_EXCEPTION_IF_NULL(stub_node);
    ClearValueMetaGradInfo(stub_node->WaitValue());
  }
}

void TopCellInfo::ResetMetaGradInfo() {
  if (meta_grad_info_.empty()) {
    return;
  }
  for (auto &item : meta_grad_info_) {
    item.first->set_auto_grad_meta_data(nullptr);
  }
  need_resume_meta_grad_ = true;
}

void TopCellInfo::ResumeMetaGradInfo() {
  if (!need_resume_meta_grad_ || meta_grad_info_.empty()) {
    return;
  }

  for (auto &item : meta_grad_info_) {
    item.first->set_auto_grad_meta_data(item.second);
  }
  need_resume_meta_grad_ = false;
}

void TopCellInfo::ClearMetaGradInfo() {
  for (auto &item : meta_grad_info_) {
    item.first->set_auto_grad_meta_data(nullptr);
  }
  meta_grad_info_.clear();
}

void TopCellInfo::Clear() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyNativeGradClearTopCell,
                                     runtime::ProfilerRecorder::kNoName, true);
  MS_LOG(DEBUG) << "Clear top cell info. Cell id " << cell_id_;
  auto_grad_cell_ptr_ = nullptr;
  is_init_kpynative_ = false;
  need_compile_graph_ = false;
  forward_already_run_ = false;
  vm_compile_ = false;
  op_index_ = 0;
  resource_ = nullptr;
  fg_ = nullptr;
  shadow_top_cell_ = nullptr;
  graph_info_map_.clear();
  replace_info_.clear();
}

void TopCellInfo::DeleteParamNodeInfo(const FuncGraphPtr &g, const std::string &id) const {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  (void)graph_info->input_params.erase(id);
}

void TopCellInfo::SetParamNodeMapInGraphInfoMap(const std::string &id, const ParameterPtr &param,
                                                bool is_weight) const {
  if (id.find('T') == std::string::npos) {
    return;
  }
  auto &graph_info = graph_info_map().at(fg());
  MS_EXCEPTION_IF_NULL(graph_info);
  if (is_weight) {
    graph_info->weight_params[id] = param;
  } else {
    graph_info->input_params[id] = param;
  }
}

void TopCellInfo::SetNodeMapInGraphInfoMap(const std::string &id, const AnfNodePtr &node, int64_t index,
                                           bool need_save_sub_id) const {
  auto &graph_info = graph_info_map().at(fg());
  MS_EXCEPTION_IF_NULL(graph_info);
  if (id.find('T') == std::string::npos) {
    return;
  }
  graph_info->node_map[id] = std::make_pair(node, std::vector<int64_t>{index});
  // For example, set id of ((A,B),C) = {CNode, -1}
  if (need_save_sub_id) {
    SetMultipleOutputToGraphInfoMap(id, node);
  }
}

void TopCellInfo::SetMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node) const {
  if (id.find("Tuple") == std::string::npos && id.find("List") == std::string::npos) {
    return;
  }
  std::vector<std::string> id_vec;
  PyNativeAlgo::Common::SplitString(id, &id_vec);
  auto tuple_size = static_cast<int64_t>(id_vec.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    // Set id of (A,B) = {CNode, 0}; Set id of C = {CNode, 1}
    SetNodeMapInGraphInfoMap(id_vec[i], node, i, false);
    SetNestedMultipleOutputToGraphInfoMap(id_vec[i], node, std::vector<int64_t>{i});
  }
}

void TopCellInfo::SetNestedMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node,
                                                        const std::vector<int64_t> &index_sequence) const {
  if (id.find("Tuple") == std::string::npos && id.find("List") == std::string::npos) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  std::vector<std::string> id_vec;
  PyNativeAlgo::Common::SplitString(id, &id_vec);
  auto tuple_size = static_cast<int64_t>(id_vec.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    std::vector<int64_t> tmp = index_sequence;
    (void)tmp.emplace_back(i);
    // Set id of A = {CNode, [0, 0]}; Set id of B = {CNode, [0, 1]};
    SetUnpackOutputToGraphInfoMap(id_vec[i], node, tmp);
    // If output have more nested tuple or list
    SetNestedMultipleOutputToGraphInfoMap(id_vec[i], node, tmp);
  }
}

void TopCellInfo::SetUnpackOutputToGraphInfoMap(const std::string &id, const AnfNodePtr &node,
                                                const std::vector<int64_t> &index) const {
  if (id.find('T') == std::string::npos) {
    return;
  }
  auto &graph_info = graph_info_map().at(fg());
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->node_map[id] = std::make_pair(node, index);
}

void TopCellInfo::SaveForwardOutputTensorInfoInBpropGraph(const FuncGraphPtr &func_graph) {
  initial_graph_param_size_ = func_graph->parameters().size();
  if (has_bprop_cut_op()) {
    MS_LOG(DEBUG) << "Top cell has bprop cut, no need to save forward output tensor info";
    return;
  }
  MS_LOG(DEBUG) << "Save top cell forward output tensor info";
  SaveForwardOutputTensorInfo(func_graph, !use_dynamic_shape_process_, &replace_info_);
}

void TopCellInfo::SetLastOutputValueForwardOutputFlag(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    const auto it = replace_info_.id_with_op_info.find(tensor->id());
    if (it != replace_info_.id_with_op_info.end()) {
      tensor->set_is_forward_output(true);
    }
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    for (const auto &v : value_seq->value()) {
      SetLastOutputValueForwardOutputFlag(v);
    }
  }
}

void TopCellInfo::ChangeTopCellInfo(const std::vector<BaseShapePtr> &args_new_shape) {
  input_args_info_->input_arg_base_shape_vec = args_new_shape;
  // Update cell id
  const auto &new_cell_id = PyNativeAlgo::Common::GetCellId(
    input_args_info_->obj_id, input_args_info_->input_arg_id_vec, input_args_info_->input_arg_value_vec);
  MS_LOG(DEBUG) << "Change top cell " << this->cell_id() << " to be unknown shape " << new_cell_id;
  cell_id_ = new_cell_id;
  input_args_info_->cell_id = new_cell_id;
  already_run_cell_id_ = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->GetAlreadyRunCellId(new_cell_id);
  MS_LOG(DEBUG) << "Get new already run top cell id " << already_run_cell_id_;
  input_args_info_->already_run_cell_id = already_run_cell_id_;
  is_unknown_shape_ = true;
}

bool TopCellInfo::IsOutputTensor(const tensor::BaseTensorPtr &tensor) const {
  return std::any_of(output_ids().begin(), output_ids().end(),
                     [&tensor](const std::string &output_id) { return tensor->id() == output_id; });
}
}  // namespace pynative
}  // namespace mindspore
