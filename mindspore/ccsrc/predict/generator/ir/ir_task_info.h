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

#ifndef MINDSPORE_MINDSPORE_CCSRC_EXECUTOR_GENERATOR_IR_IR_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_EXECUTOR_GENERATOR_IR_IR_TASK_H_
#include <cstdint>
#include <utility>
#include <memory>
#include <string>
#include <vector>
#include "proto/ge_runtime_taskinfo.pb.h"

namespace mindspore {
namespace generator {
using TaskType = ::ge::model_runner::TaskDef_TaskType;
enum TaskTmpType {
  CCE_TMP_DEF = 0,
  TBE_TMP_DEF = 1,
  AICPU_TMP_DEF = 2,
  LABEL_TMP_DEF = 3,
  EVENT_TMP_DEF = 4,
  HCCL_TMP_DEF = 5,
  PROFILER_TRACE_TMP_DEF = 6,
  MEMCPY_ASYNC_TMP_DEF = 7,
  STREAM_SWITCH_TMP_DEF = 8,
  STREAM_ACTIVE_TMP_DEF = 9
};

struct KernelContext {
  uint32_t kernel_type = 0;
  uint32_t op_id = 0;
  uint32_t kernel_func_id = 0;
  uint32_t op_index = 0;
  bool is_flowtable = false;
  std::vector<uint8_t> args_offset;
  uint32_t args_count = 0;
  std::vector<uint32_t> origin_op_index;
};

class IRtaskInfo {
 public:
  virtual ~IRtaskInfo() = default;
  virtual bool SerializeIRToProto() = 0;

 protected:
  IRtaskInfo(TaskType task_type, TaskTmpType task_tmp_type, uint64_t stream_id)
      : task_type_(task_type), task_tmp_type_(task_tmp_type), stream_id_(stream_id) {}

 public:
  uint64_t GetStreamId() const { return stream_id_; }
  TaskType GetTaskType() const { return task_type_; }
  TaskTmpType GetTaskTmpType() const { return task_tmp_type_; }

 private:
  TaskType task_type_;
  TaskTmpType task_tmp_type_;
  uint64_t stream_id_ = 0;
};

using IRtaskInfoPtr = std::shared_ptr<IRtaskInfo>;

class CceIRTaskInfo : public IRtaskInfo {
 public:
  CceIRTaskInfo(TaskType task_type, uint64_t stream_id, KernelContext k_ctx, std::string stub_func, uint32_t block_dim,
                std::vector<uint8_t> args, uint32_t args_size, std::vector<uint8_t> sm_desc,
                std::vector<uint8_t> flow_table)
      : IRtaskInfo(task_type, CCE_TMP_DEF, stream_id),
        k_ctx_(std::move(k_ctx)),
        stub_func_(std::move(stub_func)),
        block_dim_(block_dim),
        args_(std::move(args)),
        args_size_(args_size),
        sm_desc_(std::move(sm_desc)),
        flow_table_(std::move(flow_table)) {}
  ~CceIRTaskInfo() override;
  bool SerializeIRToProto() override;

 private:
  KernelContext k_ctx_;
  std::string stub_func_;
  uint32_t block_dim_ = 0;
  std::vector<uint8_t> args_;
  // uintptr_t args_addr_;
  uint32_t args_size_ = 0;
  std::vector<uint8_t> sm_desc_;
  std::vector<uint8_t> flow_table_;
};

class TbeIRTaskInfo : public IRtaskInfo {
 public:
  TbeIRTaskInfo(TaskType task_type, uint64_t stream_id, std::string stub_func, uint32_t block_dim,
                std::vector<uint8_t> args, uint32_t args_size, std::vector<uint8_t> sm_desc,
                std::vector<uint8_t> meta_data, std::vector<uintptr_t> input_data_addrs,
                std::vector<uintptr_t> output_data_addrs, std::vector<uintptr_t> workspace_addrs)
      : IRtaskInfo(task_type, TBE_TMP_DEF, stream_id),
        stub_func_(std::move(stub_func)),
        block_dim_(block_dim),
        args_(std::move(args)),
        args_size_(args_size),
        sm_desc_(std::move(sm_desc)),
        meta_data_(std::move(meta_data)),
        input_data_addrs_(std::move(input_data_addrs)),
        output_data_addrs_(std::move(output_data_addrs)),
        workspace_addrs_(std::move(workspace_addrs)) {}
  ~TbeIRTaskInfo() override;
  bool SerializeIRToProto() override;

 private:
  std::string stub_func_;
  uint32_t block_dim_ = 0;
  std::vector<uint8_t> args_;
  uint32_t args_size_ = 0;
  std::vector<uint8_t> sm_desc_;
  // uintptr_t binary_;
  // uint32_t binary_size_;
  std::vector<uint8_t> meta_data_;
  std::vector<uintptr_t> input_data_addrs_;
  std::vector<uintptr_t> output_data_addrs_;
  std::vector<uintptr_t> workspace_addrs_;
  // std::vector<uint8_t> flow_table_;
};

class AicpuIRTaskInfo : public IRtaskInfo {
 public:
  AicpuIRTaskInfo(TaskType task_type, uint64_t stream_id, std::string op_type, uint32_t flag,
                  std::vector<uint32_t> input_data_types, std::vector<std::vector<size_t>> input_data_shapes,
                  std::vector<uintptr_t> input_data_addrs, std::vector<uint32_t> output_data_types,
                  std::vector<std::vector<size_t>> output_data_shapes, std::vector<uintptr_t> output_data_addrs,
                  std::vector<uint8_t> node_def, std::vector<uint8_t> func_def)
      : IRtaskInfo(task_type, AICPU_TMP_DEF, stream_id),
        op_type_(std::move(op_type)),
        flag_(flag),
        input_data_types_(std::move(input_data_types)),
        input_data_shapes_(std::move(input_data_shapes)),
        input_data_addrs_(std::move(input_data_addrs)),
        output_data_types_(std::move(output_data_types)),
        output_data_shapes_(std::move(output_data_shapes)),
        output_data_addrs_(std::move(output_data_addrs)),
        node_def_(std::move(node_def)),
        func_def_(std::move(func_def)) {}
  ~AicpuIRTaskInfo() override;
  bool SerializeIRToProto() override;

 private:
  std::string op_type_;
  uint32_t flag_ = 0;
  std::vector<uint32_t> input_data_types_;
  std::vector<std::vector<size_t>> input_data_shapes_;
  std::vector<uintptr_t> input_data_addrs_;
  std::vector<uint32_t> output_data_types_;
  std::vector<std::vector<size_t>> output_data_shapes_;
  std::vector<uintptr_t> output_data_addrs_;
  std::vector<uint8_t> node_def_;
  std::vector<uint8_t> func_def_;
};

class LabelIRTaskInfo : public IRtaskInfo {
 public:
  LabelIRTaskInfo(TaskType task_type, uint64_t stream_id, uint32_t label_id)
      : IRtaskInfo(task_type, LABEL_TMP_DEF, stream_id), label_id_(label_id) {}
  ~LabelIRTaskInfo() override {}
  bool SerializeIRToProto() override;

 private:
  uint32_t label_id_ = 0;
};

class EventIRTaskInfo : public IRtaskInfo {
 public:
  EventIRTaskInfo(TaskType task_type, uint64_t stream_id, uint32_t event_id)
      : IRtaskInfo(task_type, EVENT_TMP_DEF, stream_id), event_id_(event_id) {}
  ~EventIRTaskInfo() override {}
  bool SerializeIRToProto() override;

 private:
  uint32_t event_id_ = 0;
};

class HcclIRTaskInfo : public IRtaskInfo {
 public:
  HcclIRTaskInfo(TaskType task_type, uint64_t stream_id, std::string hccl_type, uintptr_t input_data_addr,
                 uintptr_t output_data_addr, std::vector<uint8_t> workspace, int64_t workspace_num,
                 std::vector<uint8_t> private_def, uintptr_t ops_kernel_store, int32_t count, int64_t root_id,
                 int64_t op_type, int64_t data_type)
      : IRtaskInfo(task_type, HCCL_TMP_DEF, stream_id),
        hccl_type_(std::move(hccl_type)),
        input_data_addr_(input_data_addr),
        output_data_addr_(output_data_addr),
        workspace_(std::move(workspace)),
        workspace_num_(workspace_num),
        private_def_(std::move(private_def)),
        ops_kernel_store_(ops_kernel_store),
        count_(count),
        root_id_(root_id),
        op_type_(op_type),
        data_type_(data_type) {}
  ~HcclIRTaskInfo() override;
  bool SerializeIRToProto() override;

 private:
  std::string hccl_type_;
  uintptr_t input_data_addr_ = 0;
  uintptr_t output_data_addr_ = 0;
  std::vector<uint8_t> workspace_;
  int64_t workspace_num_ = 0;
  std::vector<uint8_t> private_def_;
  uintptr_t ops_kernel_store_ = 0;
  int32_t count_ = 0;
  int64_t root_id_ = 0;
  int64_t op_type_ = 0;
  int64_t data_type_ = 0;
};

class ProfilerIRTaskInfo : public IRtaskInfo {
 public:
  ProfilerIRTaskInfo(TaskType task_type, uint64_t stream_id, uint64_t log_id, bool notify, uint32_t flat)
      : IRtaskInfo(task_type, PROFILER_TRACE_TMP_DEF, stream_id), log_id_(log_id), notify_(notify), flat_(flat) {}
  ~ProfilerIRTaskInfo() override {}
  bool SerializeIRToProto() override;

 private:
  uint64_t log_id_ = 0;
  bool notify_ = false;
  uint32_t flat_ = 0;
};

class MemcpyAsyncIRTaskInfo : public IRtaskInfo {
 public:
  MemcpyAsyncIRTaskInfo(TaskType task_type, uint32_t stream_id, uint64_t dst, uint64_t dst_max, uint64_t src,
                        uint64_t count, int64_t kind)
      : IRtaskInfo(task_type, MEMCPY_ASYNC_TMP_DEF, stream_id),
        dst_(dst),
        dst_max_(dst_max),
        src_(src),
        count_(count),
        kind_(kind) {}
  ~MemcpyAsyncIRTaskInfo() override {}
  bool SerializeIRToProto() override;

 private:
  uint64_t dst_ = 0;
  uint64_t dst_max_ = 0;
  uint64_t src_ = 0;
  uint64_t count_ = 0;
  uint32_t kind_ = 0;
};

class StreamSwitchIRTaskInfo : public IRtaskInfo {
 public:
  StreamSwitchIRTaskInfo(TaskType task_type, uint64_t stream_id, uint32_t true_stream_id, uintptr_t input_addr,
                         uintptr_t value_addr, uint32_t cond, int64_t data_type)
      : IRtaskInfo(task_type, STREAM_SWITCH_TMP_DEF, stream_id),
        true_stream_id_(true_stream_id),
        input_addr_(input_addr),
        value_addr_(value_addr),
        cond_(cond),
        data_type_(data_type) {}
  ~StreamSwitchIRTaskInfo() override {}
  bool SerializeIRToProto() override;

 private:
  uint32_t true_stream_id_ = 0;
  uintptr_t input_addr_ = 0;
  uintptr_t value_addr_ = 0;
  uint32_t cond_ = 0;
  int64_t data_type_ = 0;
};

class StreamActiveIRTaskInfo : public IRtaskInfo {
 public:
  StreamActiveIRTaskInfo(TaskType task_type, uint64_t stream_id, uint32_t active_stream_id)
      : IRtaskInfo(task_type, STREAM_ACTIVE_TMP_DEF, stream_id), active_stream_id_(active_stream_id) {}
  ~StreamActiveIRTaskInfo() override {}
  bool SerializeIRToProto() override;

 private:
  uint32_t active_stream_id_ = 0;
};
};  // namespace generator
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_EXECUTOR_GENERATOR_IR_IR_TASK_H_
