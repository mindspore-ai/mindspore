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
#include <iostream>
#include "pipeline/pynative/pynative_utils.h"
#include "ir/tensor.h"
#include "runtime/device/device_address.h"

namespace mindspore {
namespace pynative {
namespace {
void SplitString(const std::string &str, std::vector<std::string> *id_vec) {
  constexpr char colon_delim = ':';
  constexpr char angle_bracket_left_delim = '<';
  constexpr char angle_bracket_right_delim = '>';
  auto paren_pos = str.find_first_of(angle_bracket_left_delim);
  if (paren_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Get wrong str " << str;
  }
  size_t str_size = str.size();
  const auto &sub_str = str.substr(paren_pos + 1, str_size - paren_pos - 2);
  MS_LOG(DEBUG) << "Ori str " << str << ", get sub str " << sub_str;
  auto begin = sub_str.find_first_not_of(colon_delim, 0);
  auto end = sub_str.find_first_of(colon_delim, begin);
  MS_EXCEPTION_IF_NULL(id_vec);
  while (end != std::string::npos || begin != std::string::npos) {
    const auto &sub = sub_str.substr(begin, end - begin);
    if (sub.find(angle_bracket_left_delim) != std::string::npos &&
        sub.find(angle_bracket_right_delim) == std::string::npos) {
      end = sub_str.find_first_of(angle_bracket_right_delim, begin) + 1;
      (void)id_vec->emplace_back(sub_str.substr(begin, end - begin));
    } else {
      (void)id_vec->emplace_back(sub);
    }
    begin = sub_str.find_first_not_of(colon_delim, end);
    end = sub_str.find_first_of(colon_delim, begin);
    paren_pos = sub_str.find_first_of(angle_bracket_left_delim, begin);
    if (paren_pos < end) {
      const auto &s = sub_str.substr(begin, end - begin);
      auto num = std::count(s.begin(), s.end(), angle_bracket_left_delim);
      end = begin;
      size_t last_end = 0;
      while (num--) {
        end = sub_str.find_first_of(angle_bracket_right_delim, end);
        if (last_end != 0 && end != std::string::npos && end > last_end) {
          const auto &s_t = sub_str.substr(last_end, end - last_end);
          num += std::count(s_t.begin(), s_t.end(), angle_bracket_left_delim);
        }
        last_end = end;
        ++end;
      }
      if (end >= sub_str.size()) {
        end = std::string::npos;
      }
    }
  }
}
}  // namespace

void TopCellInfo::RecordCellBackwardHookOp(const std::string &cell_order, const AnfNodePtr &hook_op) {
  MS_EXCEPTION_IF_NULL(hook_op);
  (void)cell_backward_hook_op_[cell_order].emplace_back(hook_op);
  constexpr size_t cell_backward_hook_max_num = 2;
  if (cell_backward_hook_op_[cell_order].size() > cell_backward_hook_max_num) {
    MS_LOG(EXCEPTION) << "Cell order: " << cell_order << " only has two backward hook op.";
  }
}

void TopCellInfo::GetOpInfo(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::string input_args_info;
  // Record input args info (weight or data)
  // self.p = Parameter();
  // def construct(x, y)
  //   if y:
  //        x = x + x
  //   else:
  //        x = x + self.p
  //   return x
  for (auto &t : op_run_info->base_op_run_info.input_tensor) {
    MS_EXCEPTION_IF_NULL(t);
    if (t->is_parameter() && t->param_info() != nullptr && t->param_info()->requires_grad()) {
      input_args_info += "w";
    } else {
      input_args_info += "d";
    }
  }
  // Record op name and index
  op_run_info->op_info.clear();
  op_run_info->op_info +=
    op_run_info->base_op_run_info.op_name + "-" + std::to_string(op_index_) + "-" + input_args_info;
  const auto &out_abs = op_run_info->base_op_run_info.abstract;
  MS_EXCEPTION_IF_NULL(out_abs);
  auto shape = out_abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  if (!shape->isa<abstract::NoShape>() && !shape->IsDimZero()) {
    op_run_info->op_info += "-" + shape->ToString();
  }
  op_run_info->op_index = op_index_;
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
  // Get all tensors obj in value node of running graph
  std::vector<tensor::TensorPtr> tensors_in_bprop_graph;
  MS_EXCEPTION_IF_NULL(resource_);
  const auto &bprop_graph = resource_->func_graph();
  if (bprop_graph == nullptr) {
    return;
  }
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
  hook_changed_ = false;
  is_init_kpynative_ = false;
  need_compile_graph_ = false;
  forward_already_run_ = false;
  vm_compile_ = false;
  op_index_ = 0;
  resource_ = nullptr;
  fg_ = nullptr;
  graph_info_map_.clear();
  op_info_with_tensor_id_.clear();
  tensor_id_with_tensor_object_.clear();
  op_info_with_ms_func_forward_tensors_.clear();
  cnode_hash_with_op_index_.clear();
}

void TopCellInfo::DeleteParamNodeInfo(const FuncGraphPtr &g, const std::string &id) {
  auto &graph_info = graph_info_map().at(g);
  MS_EXCEPTION_IF_NULL(graph_info);
  graph_info->input_params.erase(id);
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
  SplitString(id, &id_vec);
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
  SplitString(id, &id_vec);
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

void TopCellInfo::set_opinfo_with_tensor_id(const std::string &op_info,
                                            const std::vector<tensor::TensorPtr> &op_out_tensors) {
  if (op_info_with_tensor_id_.find(op_info) != op_info_with_tensor_id_.end()) {
    MS_LOG(EXCEPTION) << "Top cell: " << cell_id_ << " records op info with tensor id, but get op info " << op_info
                      << " in op_info_with_tensor_id map";
  }
  // Record the relationship between the forward op and its output tensor id
  for (const auto &tensor : op_out_tensors) {
    (void)op_info_with_tensor_id_[op_info].emplace_back(tensor->id());
  }
}
}  // namespace pynative
}  // namespace mindspore
