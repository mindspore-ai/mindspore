/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_INFO_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "external/graph/types.h"
#include "hccl/hccl_types.h"
#include "ir/tensor.h"

namespace mindspore::ge::model_runner {
enum TaskInfoType {
  CCE = 0,
  TBE,
  AICPU,
  LABEL_SET,
  LABEL_SWITCH,
  LABEL_GOTO,
  EVENT_RECORD,
  EVENT_WAIT,
  FUSION_START,
  FUSION_END,
  HCCL,
  PROFILER_TRACE,
  MEMCPY_ASYNC,
  STREAM_SWITCH,
  STREAM_ACTIVE,
  END_GRAPH,
  // Insert new task type here
  REVSERVED = 23
};

class TaskInfo {
 public:
  virtual ~TaskInfo() {}
  uint32_t stream_id() const { return stream_id_; }
  TaskInfoType type() const { return type_; }
  std::string op_name() const { return op_name_; }
  bool dump_flag() const { return dump_flag_; }

 protected:
  TaskInfo(const std::string &op_name, uint32_t stream_id, TaskInfoType type, bool dump_flag)
      : op_name_(op_name), stream_id_(stream_id), type_(type), dump_flag_(dump_flag) {}

 private:
  std::string op_name_;
  uint32_t stream_id_;
  TaskInfoType type_;
  bool dump_flag_;
};

class TbeTaskInfo : public TaskInfo {
 public:
  TbeTaskInfo(const std::string &op_name, uint32_t stream_id, const std::string &stub_func, uint32_t block_dim,
              const std::vector<uint8_t> &args, uint32_t args_size, const std::vector<uint8_t> &sm_desc, void *binary,
              uint32_t binary_size, const std::vector<uint8_t> &meta_data, const std::vector<void *> &input_data_addrs,
              const std::vector<void *> &output_data_addrs, const std::vector<void *> &workspace_addrs, bool dump_flag)
      : TaskInfo(op_name, stream_id, TaskInfoType::TBE, dump_flag),
        stub_func_(stub_func),
        block_dim_(block_dim),
        args_(args),
        args_size_(args_size),
        sm_desc_(sm_desc),
        binary_(binary),
        binary_size_(binary_size),
        meta_data_(meta_data),
        input_data_addrs_(input_data_addrs),
        output_data_addrs_(output_data_addrs),
        workspace_addrs_(workspace_addrs) {}
  ~TbeTaskInfo() override { binary_ = nullptr; }

  const std::string &stub_func() const { return stub_func_; }
  uint32_t block_dim() const { return block_dim_; }
  const std::vector<uint8_t> &args() const { return args_; }
  uint32_t args_size() const { return args_size_; }
  const std::vector<uint8_t> &sm_desc() const { return sm_desc_; }
  void *binary() const { return binary_; }
  uint32_t binary_size() const { return binary_size_; }
  const std::vector<uint8_t> &meta_data() const { return meta_data_; }
  const std::vector<void *> &input_data_addrs() const { return input_data_addrs_; }
  const std::vector<void *> &output_data_addrs() const { return output_data_addrs_; }
  const std::vector<void *> &workspace_addrs() const { return workspace_addrs_; }

  void SetBinary(void *binary, uint32_t binary_size) {
    binary_ = binary;
    binary_size_ = binary_size;
  }

 private:
  std::string stub_func_;
  uint32_t block_dim_;
  std::vector<uint8_t> args_;
  uint32_t args_size_;
  std::vector<uint8_t> sm_desc_;
  void *binary_;
  uint32_t binary_size_;
  std::vector<uint8_t> meta_data_;
  std::vector<void *> input_data_addrs_;
  std::vector<void *> output_data_addrs_;
  std::vector<void *> workspace_addrs_;
};

class AicpuTaskInfo : public TaskInfo {
 public:
  AicpuTaskInfo(const std::string &op_name, uint32_t stream_id, const std::string &so_name,
                const std::string &kernel_name, const std::string &node_def, const std::string &ext_info,
                const std::vector<void *> &input_data_addrs, const std::vector<void *> &output_data_addrs,
                bool dump_flag, bool cust_aicpu = false, bool is_blocking = false, uint32_t ms_event_id = 0,
                ::ge::UnknowShapeOpType unknow_type = ::ge::UnknowShapeOpType::DEPEND_IN_SHAPE)
      : TaskInfo(op_name, stream_id, TaskInfoType::AICPU, dump_flag),
        so_name_(so_name),
        kernel_name_(kernel_name),
        node_def_(node_def),
        ext_info_(ext_info),
        input_data_addrs_(input_data_addrs),
        output_data_addrs_(output_data_addrs),
        cust_aicpu_(cust_aicpu),
        is_blocking_(is_blocking),
        ms_event_id_(ms_event_id),
        unknow_type_(unknow_type) {}
  ~AicpuTaskInfo() override {}

  const std::string &so_name() const { return so_name_; }
  const std::string &kernel_name() const { return kernel_name_; }
  const std::string &node_def() const { return node_def_; }
  const std::vector<void *> &input_data_addrs() const { return input_data_addrs_; }
  const std::vector<void *> &output_data_addrs() const { return output_data_addrs_; }
  const std::string &ext_info() const { return ext_info_; }
  const bool cust_aicpu() const { return cust_aicpu_; }
  const bool is_blocking() const { return is_blocking_; }
  const uint32_t ms_event_id() const { return ms_event_id_; }
  const ::ge::UnknowShapeOpType unknown_type() const { return unknow_type_; }

 private:
  std::string so_name_;
  std::string kernel_name_;
  std::string node_def_;
  std::string ext_info_;
  std::vector<void *> input_data_addrs_;
  std::vector<void *> output_data_addrs_;
  bool cust_aicpu_;
  // if true, means the op is async, and need FWK_ADPT_EXT_ASYNCWAIT in ext_info and UpdateEventId (GetNext).
  bool is_blocking_;
  uint32_t ms_event_id_;
  ::ge::UnknowShapeOpType unknow_type_;
};

class LabelSetTaskInfo : public TaskInfo {
 public:
  LabelSetTaskInfo(const std::string &op_name, uint32_t stream_id, uint32_t label_id)
      : TaskInfo(op_name, stream_id, TaskInfoType::LABEL_SET, false), label_id_(label_id) {}
  ~LabelSetTaskInfo() override {}
  uint32_t label_id() const { return label_id_; }

 private:
  uint32_t label_id_;
};

class LabelGotoTaskInfo : public TaskInfo {
 public:
  LabelGotoTaskInfo(const std::string &op_name, uint32_t stream_id, uint32_t label_id)
      : TaskInfo(op_name, stream_id, TaskInfoType::LABEL_GOTO, false), label_id_(label_id) {}
  ~LabelGotoTaskInfo() override {}
  uint32_t label_id() const { return label_id_; }

 private:
  uint32_t label_id_;
};

class LabelSwitchTaskInfo : public TaskInfo {
 public:
  LabelSwitchTaskInfo(const std::string &op_name, uint32_t stream_id, uint32_t label_size,
                      const std::vector<uint32_t> &label_list, void *cond)
      : TaskInfo(op_name, stream_id, TaskInfoType::LABEL_SWITCH, false),
        label_size_(label_size),
        label_list_(label_list),
        cond_(cond) {}
  ~LabelSwitchTaskInfo() override { cond_ = nullptr; }
  uint32_t label_size() const { return label_size_; }
  const std::vector<uint32_t> &label_list() const { return label_list_; }
  void *cond() const { return cond_; }

 private:
  uint32_t label_size_;
  std::vector<uint32_t> label_list_;
  void *cond_;
};

class EventTaskInfo : public TaskInfo {
 public:
  uint32_t event_id() const { return event_id_; }

 protected:
  EventTaskInfo(const std::string &op_name, uint32_t stream_id, TaskInfoType type, uint32_t event_id)
      : TaskInfo(op_name, stream_id, type, false), event_id_(event_id) {}
  ~EventTaskInfo() override {}

  uint32_t event_id_;
};

class EventRecordTaskInfo : public EventTaskInfo {
 public:
  EventRecordTaskInfo(const std::string &op_name, uint32_t stream_id, uint32_t event_id)
      : EventTaskInfo(op_name, stream_id, TaskInfoType::EVENT_RECORD, event_id) {}
  ~EventRecordTaskInfo() override {}
};

class EventWaitTaskInfo : public EventTaskInfo {
 public:
  EventWaitTaskInfo(const std::string &op_name, uint32_t stream_id, uint32_t event_id)
      : EventTaskInfo(op_name, stream_id, TaskInfoType::EVENT_WAIT, event_id) {}
  ~EventWaitTaskInfo() override {}
};

class FusionStartTaskInfo : public TaskInfo {
 public:
  explicit FusionStartTaskInfo(const std::string &op_name, uint32_t stream_id)
      : TaskInfo(op_name, stream_id, TaskInfoType::FUSION_START, false) {}
  ~FusionStartTaskInfo() override {}
};

class FusionEndTaskInfo : public TaskInfo {
 public:
  explicit FusionEndTaskInfo(const std::string &op_name, uint32_t stream_id)
      : TaskInfo(op_name, stream_id, TaskInfoType::FUSION_END, false) {}
  ~FusionEndTaskInfo() override {}
};

class HcclTaskInfo : public TaskInfo {
 public:
  HcclTaskInfo(const std::string &op_name, uint32_t stream_id, const std::string hccl_type, void *input_data_addr,
               void *output_data_addr, void *workspace_addr, int64_t workspace_size, int64_t hccl_stream_num,
               const std::vector<uint8_t> &private_def, void *ops_kernel_store, int32_t count, int64_t root_id,
               int64_t op_type, int64_t data_type, const std::string &group, bool dump_flag)
      : TaskInfo(op_name, stream_id, TaskInfoType::HCCL, dump_flag),
        hccl_type_(hccl_type),
        input_data_addr_(input_data_addr),
        output_data_addr_(output_data_addr),
        workspace_addr_(workspace_addr),
        workspace_size_(workspace_size),
        hccl_stream_num_(hccl_stream_num),
        private_def_(private_def),
        ops_kernel_store_(ops_kernel_store),
        count_(count),
        root_id_(root_id),
        op_type_(op_type),
        data_type_(data_type),
        group_(group) {}
  ~HcclTaskInfo() override {}

  const std::string &hccl_type() const { return hccl_type_; }
  void *input_data_addr() const { return input_data_addr_; }
  void *output_data_addr() const { return output_data_addr_; }
  void *workspace_addr() const { return workspace_addr_; }
  int64_t workspace_size() const { return workspace_size_; }
  int64_t hccl_stream_num() const { return hccl_stream_num_; }
  const std::vector<uint8_t> &private_def() const { return private_def_; }
  void *ops_kernel_store() const { return ops_kernel_store_; }
  int32_t count() const { return count_; }
  int64_t root_id() const { return root_id_; }
  uint32_t graph_id() const { return graph_id_; }
  void set_graph_id(uint32_t graph_id) { graph_id_ = graph_id; }
  int64_t op_type() const { return op_type_; }
  int64_t data_type() const { return data_type_; }
  size_t output_num() const { return output_num_; }
  void set_output_num(size_t output_num) { output_num_ = output_num; }
  std::vector<std::string> data_format() const { return data_format_; }
  void add_data_format(const std::string &data_format) { data_format_.push_back(data_format); }
  std::vector<std::vector<int64_t>> hccl_kernel_output_shape_list() const { return hccl_kernel_output_shape_list_; }
  void set_hccl_kernel_output_shape_list(const std::vector<std::vector<int64_t>> &hccl_kernel_output_shape_list) {
    hccl_kernel_output_shape_list_ = hccl_kernel_output_shape_list;
  }
  std::vector<std::vector<int64_t>> hccl_host_output_shape_list() const { return hccl_host_output_shape_list_; }
  void set_hccl_host_output_shape_list(const std::vector<std::vector<int64_t>> &hccl_host_output_shape_list) {
    hccl_host_output_shape_list_ = hccl_host_output_shape_list;
  }
  std::vector<size_t> output_size_list() const { return output_size_list_; }
  void add_output_size_list(size_t output_size) { output_size_list_.push_back(output_size); }
  std::map<std::string, tensor::TensorPtr> device_loop_control_tensors() const { return device_loop_ctrl_tensors_; }
  void set_device_loop_ctrl_tensors(const std::map<std::string, tensor::TensorPtr> &device_loop_ctrl_tensors) {
    device_loop_ctrl_tensors_ = device_loop_ctrl_tensors;
  }
  std::vector<const void *> get_output_addr_list() const { return output_addr_list_; }
  void add_output_addr(const void *output_addr) { output_addr_list_.push_back(output_addr); }
  const std::string &group() const { return group_; }
  const std::vector<void *> &global_workspace_addr() const { return global_workspace_addr_; }

  void SetGlobalWorkspaceAddr(const std::vector<void *> &global_workspace_addr) {
    this->global_workspace_addr_ = global_workspace_addr;
  }

 private:
  std::string hccl_type_;
  void *input_data_addr_;
  void *output_data_addr_;
  std::vector<const void *> output_addr_list_;
  void *workspace_addr_;
  int64_t workspace_size_;
  int64_t hccl_stream_num_;
  std::vector<uint8_t> private_def_;
  void *ops_kernel_store_;
  int32_t count_;
  int64_t root_id_;
  uint32_t graph_id_;
  int64_t op_type_;
  int64_t data_type_;
  std::string group_;
  size_t output_num_;
  std::vector<std::string> data_format_;
  std::vector<std::vector<int64_t>> hccl_kernel_output_shape_list_;
  std::vector<std::vector<int64_t>> hccl_host_output_shape_list_;
  std::vector<size_t> output_size_list_;
  std::map<std::string, tensor::TensorPtr> device_loop_ctrl_tensors_;
  // hccl global overflow addr
  std::vector<void *> global_workspace_addr_;
};

class ProfilerTraceTaskInfo : public TaskInfo {
 public:
  ProfilerTraceTaskInfo(const std::string &op_name, uint32_t stream_id, uint64_t log_id, bool notify, uint32_t flat)
      : TaskInfo(op_name, stream_id, TaskInfoType::PROFILER_TRACE, false),
        log_id_(log_id),
        notify_(notify),
        flat_(flat) {}
  ~ProfilerTraceTaskInfo() override {}

  uint64_t log_id() const { return log_id_; }
  bool notify() const { return notify_; }
  uint32_t flat() const { return flat_; }

 private:
  uint64_t log_id_;
  bool notify_;
  uint32_t flat_;
};

class MemcpyAsyncTaskInfo : public TaskInfo {
 public:
  MemcpyAsyncTaskInfo(const std::string &op_name, uint32_t stream_id, void *dst, uint64_t dst_max, void *src,
                      uint64_t count, uint32_t kind, bool dump_flag)
      : TaskInfo(op_name, stream_id, TaskInfoType::MEMCPY_ASYNC, dump_flag),
        dst_(dst),
        dst_max_(dst_max),
        src_(src),
        count_(count),
        kind_(kind) {}
  ~MemcpyAsyncTaskInfo() override {}

  void *dst() const { return dst_; }
  uint64_t dst_max() const { return dst_max_; }
  void *src() const { return src_; }
  uint64_t count() const { return count_; }
  uint32_t kind() const { return kind_; }

 private:
  void *dst_;
  uint64_t dst_max_;
  void *src_;
  uint64_t count_;
  uint32_t kind_;
};

class EndGraphTaskInfo : public TaskInfo {
 public:
  EndGraphTaskInfo(const std::string &op_name, uint32_t stream_id, bool dump_flag)
      : TaskInfo(op_name, stream_id, TaskInfoType::END_GRAPH, dump_flag) {}
  ~EndGraphTaskInfo() override {}
};

class StreamSwitchTaskInfo : public TaskInfo {
 public:
  StreamSwitchTaskInfo(const std::string &op_name, uint32_t stream_id, int64_t true_stream_id, void *input_addr,
                       void *value_addr, int64_t cond, int64_t data_type)
      : TaskInfo(op_name, stream_id, TaskInfoType::STREAM_SWITCH, false),
        true_stream_id_(true_stream_id),
        input_addr_(input_addr),
        value_addr_(value_addr),
        cond_(cond),
        data_type_(data_type) {}
  ~StreamSwitchTaskInfo() override {
    input_addr_ = nullptr;
    value_addr_ = nullptr;
  }

  int64_t true_stream_id() const { return true_stream_id_; }
  void *input_addr() const { return input_addr_; }
  void *value_addr() const { return value_addr_; }
  int64_t cond() const { return cond_; }
  int64_t data_type() const { return data_type_; }

 private:
  int64_t true_stream_id_;
  void *input_addr_;
  void *value_addr_;
  int64_t cond_;
  int64_t data_type_;
};

class StreamActiveTaskInfo : public TaskInfo {
 public:
  StreamActiveTaskInfo(const std::string &op_name, uint32_t stream_id, uint32_t active_stream_id)
      : TaskInfo(op_name, stream_id, TaskInfoType::STREAM_ACTIVE, false), active_stream_id_(active_stream_id) {}
  ~StreamActiveTaskInfo() override {}

  uint32_t active_stream_id() const { return active_stream_id_; }

 private:
  uint32_t active_stream_id_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_INFO_H_
