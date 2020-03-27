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

#include "predict/generator/ir/ir_task_info.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace generator {
bool CceIRTaskInfo::SerializeIRToProto() {
  auto cce_task_def_ptr = std::unique_ptr<ge::model_runner::CceTaskDef>();
  auto kernel_context_ptr = std::unique_ptr<ge::model_runner::KernelContext>();
  MS_EXCEPTION_IF_NULL(cce_task_def_ptr);
  MS_EXCEPTION_IF_NULL(kernel_context_ptr);
  kernel_context_ptr->set_kernel_type(k_ctx_.kernel_type);
  kernel_context_ptr->set_op_id(k_ctx_.op_id);
  kernel_context_ptr->set_kernel_func_id(k_ctx_.kernel_func_id);
  kernel_context_ptr->set_op_index(k_ctx_.op_index);
  kernel_context_ptr->set_is_flowtable(k_ctx_.is_flowtable);
  kernel_context_ptr->set_args_count(k_ctx_.args_count);
  for (unsigned int i : k_ctx_.origin_op_index) {
    kernel_context_ptr->add_origin_op_index(i);
  }
  void *tmp_args_offset = static_cast<void *>((k_ctx_.args_offset).data());
  if (tmp_args_offset == nullptr) {
    MS_LOG(WARNING) << "tmp_args_offset have no data";
    return false;
  }
  kernel_context_ptr->set_args_offset(tmp_args_offset, k_ctx_.args_offset.size());
  cce_task_def_ptr->set_allocated_kernel_context(std::move(kernel_context_ptr).get());
  cce_task_def_ptr->set_stub_func(stub_func_);
  cce_task_def_ptr->set_block_dim(block_dim_);
  cce_task_def_ptr->set_args_size(args_size_);
  void *tmp_sm_desc = static_cast<void *>(sm_desc_.data());
  if (tmp_sm_desc == nullptr) {
    MS_LOG(WARNING) << "tmp_sm_desc have no data";
    return false;
  }
  cce_task_def_ptr->set_sm_desc(tmp_sm_desc, sm_desc_.size());

  void *tmp_flow_table = static_cast<void *>(flow_table_.data());
  if (tmp_flow_table == nullptr) {
    MS_LOG(WARNING) << "tmp_flow_table have no data";
    return false;
  }
  cce_task_def_ptr->set_flow_table(tmp_flow_table, flow_table_.size());
  return true;
}

CceIRTaskInfo::~CceIRTaskInfo() {
  args_.clear();
  sm_desc_.clear();
  flow_table_.clear();
}

bool TbeIRTaskInfo::SerializeIRToProto() {
  auto tbe_task_def_ptr = std::unique_ptr<ge::model_runner::TbeTaskDef>();
  MS_EXCEPTION_IF_NULL(tbe_task_def_ptr);
  tbe_task_def_ptr->set_stub_func(stub_func_);
  tbe_task_def_ptr->set_block_dim(block_dim_);
  tbe_task_def_ptr->set_args_size(args_size_);
  void *tmp_args = static_cast<void *>(args_.data());
  if (tmp_args == nullptr) {
    MS_LOG(WARNING) << "tmp_args have no data";
    return false;
  }
  tbe_task_def_ptr->set_args(tmp_args, args_.size());
  void *tmp_sm_desc = static_cast<void *>(sm_desc_.data());
  if (tmp_sm_desc == nullptr) {
    MS_LOG(WARNING) << "tmp_sm_desc have no data";
    return false;
  }
  tbe_task_def_ptr->set_sm_desc(tmp_sm_desc, sm_desc_.size());
  void *tmp_meta_data = static_cast<void *>(meta_data_.data());
  if (tmp_meta_data == nullptr) {
    MS_LOG(WARNING) << "tmp_meta_data have no data";
    return false;
  }
  tbe_task_def_ptr->set_meta_data(tmp_meta_data, meta_data_.size());
  for (auto &in : input_data_addrs_) {
    tbe_task_def_ptr->add_input_addrs(in);
  }
  for (auto &ou : output_data_addrs_) {
    tbe_task_def_ptr->add_output_addrs(ou);
  }
  for (auto &wk : workspace_addrs_) {
    tbe_task_def_ptr->add_workspace_addrs(wk);
  }
  return true;
}

TbeIRTaskInfo::~TbeIRTaskInfo() {
  args_.clear();
  sm_desc_.clear();
  meta_data_.clear();
  input_data_addrs_.clear();
  output_data_addrs_.clear();
  workspace_addrs_.clear();
}

bool AicpuIRTaskInfo::SerializeIRToProto() {
  auto aicpu_task_def_ptr = std::unique_ptr<ge::model_runner::AicpuTaskDef>();
  MS_EXCEPTION_IF_NULL(aicpu_task_def_ptr);
  aicpu_task_def_ptr->set_op_type(op_type_);
  aicpu_task_def_ptr->set_flag(flag_);
  for (auto &shape : input_data_shapes_) {
    auto in_shape_ptr = aicpu_task_def_ptr->add_input_shapes();
    for (auto &in_sh : shape) {
      in_shape_ptr->add_shape(static_cast<uint32_t>(in_sh));
    }
  }
  for (auto &shape : output_data_shapes_) {
    auto ou_shape_ptr = aicpu_task_def_ptr->add_output_shapes();
    for (auto &ou_sh : shape) {
      ou_shape_ptr->add_shape(static_cast<uint32_t>(ou_sh));
    }
  }
  for (auto &in_type : input_data_types_) {
    aicpu_task_def_ptr->add_input_types(in_type);
  }
  for (auto &ou_type : output_data_types_) {
    aicpu_task_def_ptr->add_output_types(ou_type);
  }
  for (auto &in_addr : input_data_addrs_) {
    aicpu_task_def_ptr->add_input_addrs(in_addr);
  }
  for (auto &ou_addr : output_data_addrs_) {
    aicpu_task_def_ptr->add_output_addrs(ou_addr);
  }
  void *tmp_node_def = static_cast<void *>(node_def_.data());
  if (tmp_node_def == nullptr) {
    MS_LOG(WARNING) << "tmp_node_def have no data";
    return false;
  }
  aicpu_task_def_ptr->set_node_def(tmp_node_def, node_def_.size());
  void *tmp_func_def = static_cast<void *>(func_def_.data());
  if (tmp_func_def == nullptr) {
    MS_LOG(WARNING) << "tmp_func_def have no data";
    return false;
  }
  aicpu_task_def_ptr->set_func_def(tmp_func_def, func_def_.size());
  return true;
}

AicpuIRTaskInfo::~AicpuIRTaskInfo() {
  input_data_types_.clear();
  input_data_shapes_.clear();
  input_data_addrs_.clear();
  output_data_types_.clear();
  output_data_shapes_.clear();
  output_data_addrs_.clear();
  node_def_.clear();
  func_def_.clear();
}

bool LabelIRTaskInfo::SerializeIRToProto() {
  auto label_task_def_ptr = std::unique_ptr<ge::model_runner::LabelTaskDef>();
  MS_EXCEPTION_IF_NULL(label_task_def_ptr);
  label_task_def_ptr->set_label_id(label_id_);
  return true;
}

bool EventIRTaskInfo::SerializeIRToProto() {
  auto event_task_def_ptr = std::unique_ptr<ge::model_runner::EventTaskDef>();
  MS_EXCEPTION_IF_NULL(event_task_def_ptr);
  event_task_def_ptr->set_event_id(event_id_);
  return true;
}

bool HcclIRTaskInfo::SerializeIRToProto() {
  auto hccl_task_def_ptr = std::unique_ptr<ge::model_runner::HcclTaskDef>();
  MS_EXCEPTION_IF_NULL(hccl_task_def_ptr);
  hccl_task_def_ptr->set_hccl_type(hccl_type_);
  hccl_task_def_ptr->set_input_addr(input_data_addr_);
  hccl_task_def_ptr->set_output_addr(output_data_addr_);
  auto tmp_wk = static_cast<void *>(workspace_.data());
  hccl_task_def_ptr->set_workspace(tmp_wk, workspace_.size());
  hccl_task_def_ptr->set_workspace_num(workspace_num_);
  auto tmp_pri_def = static_cast<void *>(private_def_.data());
  hccl_task_def_ptr->set_private_def(tmp_pri_def, private_def_.size());
  hccl_task_def_ptr->set_ops_kernel_store(ops_kernel_store_);
  hccl_task_def_ptr->set_count(count_);
  hccl_task_def_ptr->set_root_id(root_id_);
  hccl_task_def_ptr->set_op_type(op_type_);
  hccl_task_def_ptr->set_data_type(data_type_);
  return true;
}

HcclIRTaskInfo::~HcclIRTaskInfo() {
  workspace_.clear();
  private_def_.clear();
}

bool ProfilerIRTaskInfo::SerializeIRToProto() {
  auto profiler_task_def_ptr = std::unique_ptr<ge::model_runner::ProfilerTaskDef>();
  MS_EXCEPTION_IF_NULL(profiler_task_def_ptr);
  profiler_task_def_ptr->set_log_id(log_id_);
  profiler_task_def_ptr->set_flat(flat_);
  profiler_task_def_ptr->set_notify(notify_);
  return true;
}

bool MemcpyAsyncIRTaskInfo::SerializeIRToProto() {
  auto mem_task_def_ptr = std::unique_ptr<ge::model_runner::MemcpyAsyncTaskDef>();
  MS_EXCEPTION_IF_NULL(mem_task_def_ptr);
  mem_task_def_ptr->set_dst(dst_);
  mem_task_def_ptr->set_dst_max(dst_max_);
  mem_task_def_ptr->set_src(src_);
  mem_task_def_ptr->set_count(count_);
  mem_task_def_ptr->set_kind(kind_);
  return true;
}

bool StreamSwitchIRTaskInfo::SerializeIRToProto() {
  auto stream_switch_task_def_ptr = std::unique_ptr<ge::model_runner::StreamSwitchTaskDef>();
  MS_EXCEPTION_IF_NULL(stream_switch_task_def_ptr);
  stream_switch_task_def_ptr->set_true_stream_id(true_stream_id_);
  stream_switch_task_def_ptr->set_input_addr(input_addr_);
  stream_switch_task_def_ptr->set_value_addr(value_addr_);
  stream_switch_task_def_ptr->set_cond(cond_);
  stream_switch_task_def_ptr->set_data_type(data_type_);
  return true;
}

bool StreamActiveIRTaskInfo::SerializeIRToProto() {
  auto stream_active_task_def_ptr = std::unique_ptr<ge::model_runner::StreamActiveTaskDef>();
  MS_EXCEPTION_IF_NULL(stream_active_task_def_ptr);
  stream_active_task_def_ptr->set_active_stream_id(active_stream_id_);
  return true;
}
}  // namespace generator
}  // namespace mindspore
