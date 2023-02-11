/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/dump/data_dumper.h"
#include <utility>
#include <map>
#include <memory>
#include <string>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/convert_utils_base.h"
#include "runtime/dev.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "runtime/rt_model.h"
#include "plugin/device/ascend/hal/device/ge_types_convert.h"
#include "proto/op_mapping_info.pb.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "plugin/device/ascend/hal/device/dump/dumper_base.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
DataDumper::~DataDumper() {
  kernel_graph_ = nullptr;
  ReleaseDevMem(&dev_load_mem_);
  ReleaseDevMem(&dev_unload_mem_);
  ReleaseDevMem(&op_debug_buffer_addr_);
  ReleaseDevMem(&op_debug_dump_args_);
}

#ifndef ENABLE_SECURITY
void DataDumper::GetNeedDumpKernelList(NotNull<std::map<std::string, CNodePtr> *> kernel_map) const {
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  for (const auto &kernel : kernel_graph_->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelType(kernel) == HCCL_KERNEL &&
        DumpJsonParser::GetInstance().NeedDump(kernel->fullname_with_scope())) {
      auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_size; ++i) {
        auto input_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, i);
        auto input = input_with_index.first;
        MS_EXCEPTION_IF_NULL(input);
        if (input->isa<CNode>()) {
          MS_LOG(INFO) << "[AsyncDump] Match Hccl Node:" << kernel->fullname_with_scope()
                       << " Input:" << input->fullname_with_scope();
          auto it = kernel_map->try_emplace(input->fullname_with_scope(), input->cast<CNodePtr>());
          if (!it.second) {
            MS_LOG(INFO) << "Node name already exist: " << input->fullname_with_scope();
          }
        }
      }
    } else if (KernelNeedDump(kernel)) {
      MS_LOG(INFO) << "[AsyncDump] Match Node:" << kernel->fullname_with_scope();
      auto it = kernel_map->try_emplace(kernel->fullname_with_scope(), kernel);
      if (!it.second) {
        MS_LOG(INFO) << "Node name already exist: " << kernel->fullname_with_scope();
      }
    }
  }
}

void DataDumper::LoadDumpInfo() {
  MS_LOG(INFO) << "[DataDump] LoadDumpInfo start";
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  aicpu::dump::OpMappingInfo dump_info;
  SetOpEndgraphMappingInfo(NOT_NULL(&dump_info));
  SetOpDebugMappingInfo(NOT_NULL(&dump_info), debug_task_id_, debug_stream_id_, op_debug_dump_args_);
  SetOpMappingInfo(NOT_NULL(&dump_info));

  auto kernels = kernel_graph_->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (!KernelNeedDump(kernel)) {
      continue;
    }
    if (common::AnfAlgo::IsNonTaskOp(kernel)) {
      MS_LOG(INFO) << "[DataDump] kernel [" << kernel->UniqueName() << "] is a non-task node, skip dump.";
      continue;
    }
    MS_LOG(INFO) << "[DataDump] LoadDumpInfo kernel:" << kernel->UniqueName();
    (void)dump_kernel_names_.emplace_back(kernel->UniqueName());
    DumpJsonParser::GetInstance().MatchKernel(kernel->fullname_with_scope());
    if (AnfAlgo::GetKernelType(kernel) == HCCL_KERNEL) {
      MS_LOG(INFO) << "Skip HCCL kernel.";
      continue;
    }
    aicpu::dump::Task task;
    ConstructDumpTask(NOT_NULL(kernel), NOT_NULL(&task));
    MS_EXCEPTION_IF_NULL(dump_info.mutable_task());
    dump_info.mutable_task()->Add(std::move(task));
  }
  RtLoadDumpData(dump_info, &dev_load_mem_);
  load_flag_ = true;
  // graph id may changed in Unload
  graph_id_ = kernel_graph_->graph_id();
  MS_LOG(INFO) << "[DataDump] LoadDumpInfo end";
}

void DataDumper::SetOpMappingInfo(NotNull<aicpu::dump::OpMappingInfo *> dump_info) const {
  MS_LOG(INFO) << "SetOpMappinglnfo Start.";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  auto dump_path = DumpJsonParser::GetInstance().path();
  auto input_ctrl_tensors = kernel_graph_->device_loop_control_tensors();
  constexpr size_t kLoopSinkCtrlTensorNum = 5;  // cur step, next step, cur epoch, one, steps per epoch
  bool valid_ctrl_tensors = input_ctrl_tensors.size() >= kLoopSinkCtrlTensorNum;
  std::string net_name = DumpJsonParser::GetInstance().net_name();
  std::string iteration = DumpJsonParser::GetInstance().iteration_string();

  if (dump_path.empty()) {
    MS_LOG(EXCEPTION) << "Dump path invalid";
  }
  uint32_t graph_id = kernel_graph_->graph_id();
  uint32_t rank_id = 0;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    // get actual rank id if it's distribution training case.
    if (!CommManager::GetInstance().GetRankID(kHcclWorldGroup, &rank_id)) {
      MS_LOG(INFO) << "Failed to get rank id.";
    }
  }
  dump_info->set_dump_path("/" + dump_path + "/rank_" + std::to_string(rank_id) + "/");
  MS_LOG(INFO) << "[DataDump] dump_path: " << dump_path;

  dump_info->set_model_name(net_name);
  MS_LOG(INFO) << "[DataDump] model_name: " << net_name;

  MS_LOG(INFO) << "[DataDump] iteration_pre: " << iteration;
  if (iteration == "all") {
    iteration = "0-" + std::to_string(ULONG_MAX);
  }
  MS_LOG(INFO) << "[DataDump] iteration_post: " << iteration;
  dump_info->set_dump_step(iteration);

  dump_info->set_model_id(graph_id);
  dump_info->set_flag(kAicpuLoadFlag);

  if (!valid_ctrl_tensors) {
    MS_LOG(INFO) << "[DataDump] input_ctrl_tensors not valid.";
    return;
  }
  const auto &current_step_tensor = input_ctrl_tensors[kCurLoopCountName];
  const auto &current_epoch_tensor = input_ctrl_tensors[kCurEpochCountName];
  const auto &steps_per_epoch_tensor = input_ctrl_tensors[kConstLoopNumInEpochName];

  MS_EXCEPTION_IF_NULL(current_step_tensor);
  MS_EXCEPTION_IF_NULL(current_epoch_tensor);
  MS_EXCEPTION_IF_NULL(steps_per_epoch_tensor);
  MS_EXCEPTION_IF_NULL(current_step_tensor->device_address());
  MS_EXCEPTION_IF_NULL(current_epoch_tensor->device_address());
  MS_EXCEPTION_IF_NULL(steps_per_epoch_tensor->device_address());

  void *current_step = current_step_tensor->device_address()->GetMutablePtr();
  void *current_epoch = current_epoch_tensor->device_address()->GetMutablePtr();
  void *steps_per_epoch = steps_per_epoch_tensor->device_address()->GetMutablePtr();

  if (current_epoch != nullptr && current_step != nullptr && steps_per_epoch != nullptr) {
    dump_info->set_step_id_addr(reinterpret_cast<uint64_t>(current_epoch));
    dump_info->set_loop_cond_addr(reinterpret_cast<uint64_t>(current_step));
    dump_info->set_iterations_per_loop_addr(reinterpret_cast<uint64_t>(steps_per_epoch));
  } else {
    MS_LOG(INFO) << "Invalid ctrl tensor device address";
  }
  MS_LOG(INFO) << "SetOpMappinglnfo End.";
}
#endif

void DataDumper::UnloadDumpInfo() {
  if (!load_flag_) {
    MS_LOG(WARNING) << "[DataDump] Load not success, no need to unload";
    return;
  }
  MS_LOG(INFO) << "[DataDump] UnloadDumpInfo start. graphId:" << graph_id_;

  aicpu::dump::OpMappingInfo op_mapping_info;
  op_mapping_info.set_model_id(graph_id_);
  op_mapping_info.set_flag(kAicpuUnloadFlag);

  for (const auto &kernel_name : dump_kernel_names_) {
    aicpu::dump::Task task;
    std::map<std::string, std::shared_ptr<RuntimeInfo>>::const_iterator iter = runtime_info_map_.find(kernel_name);
    if (iter == runtime_info_map_.end()) {
      MS_LOG(EXCEPTION) << "[DataDump] kernel name not found in runtime_info_map";
    }
    MS_EXCEPTION_IF_NULL(iter->second);
    auto task_id = std::get<kTupleTaskId>(*iter->second);
    task.set_task_id(task_id);
    MS_EXCEPTION_IF_NULL(op_mapping_info.mutable_task());
    op_mapping_info.mutable_task()->Add(std::move(task));
  }

  RtLoadDumpData(op_mapping_info, &dev_unload_mem_);
}

void DataDumper::ConstructDumpTask(NotNull<const CNodePtr &> kernel, NotNull<aicpu::dump::Task *> dump_task) const {
  dump_task->set_end_graph(false);
  auto iter = runtime_info_map_.find(kernel->UniqueName());
  if (iter == runtime_info_map_.end()) {
    MS_LOG(EXCEPTION) << "[DataDump] kernel name not found in runtime_info_map, kernel name: " << kernel->UniqueName();
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  auto task_id = std::get<kTupleTaskId>(*iter->second);
  auto stream_id = std::get<kTupleStreamId>(*iter->second);
#ifndef ENABLE_SECURITY
  auto args = std::get<kTupleArgs>(*iter->second);
#endif
  MS_LOG(INFO) << "[DataDump] Get runtime info task_id:" << task_id << " stream_id:" << stream_id;

  dump_task->set_task_id(task_id);
  dump_task->set_stream_id(stream_id);
  MS_EXCEPTION_IF_NULL(dump_task->mutable_op());
  dump_task->mutable_op()->set_op_name(kernel->fullname_with_scope());
  dump_task->mutable_op()->set_op_type(common::AnfAlgo::GetCNodeName(kernel.get()));

#ifndef ENABLE_SECURITY
  DumpKernelOutput(kernel, args, dump_task);
  DumpKernelInput(kernel, args, dump_task);
#endif
}

void DataDumper::SetOpEndgraphMappingInfo(const NotNull<aicpu::dump::OpMappingInfo *> dump_info) const {
  for (auto &[task_id, stream_id] : end_graph_info_map_) {
    MS_LOG(INFO) << "[DataDump] Add op end graph info to OpMappingInfo, task id = " << task_id
                 << ", stream id = " << stream_id;
    aicpu::dump::Task task;
    task.set_end_graph(true);
    task.set_task_id(task_id);
    task.set_stream_id(stream_id);
    MS_EXCEPTION_IF_NULL(task.mutable_op());
    task.mutable_op()->set_op_name(kNodeNameEndGraph);
    task.mutable_op()->set_op_type(kOpTypeOpEndGraph);
    dump_info->mutable_task()->Add(std::move(task));
  }
}

#ifndef ENABLE_SECURITY
void DataDumper::OpDebugRegister() {
  uint32_t op_debug_mode = DumpJsonParser::GetInstance().op_debug_mode();
  auto iter = kOverflowModeStr.find(op_debug_mode);
  if (iter == kOverflowModeStr.end()) {
    MS_LOG(EXCEPTION) << "Invalid op debug mode " << op_debug_mode;
  }
  MS_LOG(INFO) << "[DataDump] Op debug mode is " << iter->second;
  if (op_debug_mode == kNoOverflow) {
    return;
  }

  is_op_debug_ = true;
  int64_t value = 0;
  rtError_t rt_ret = rtGetRtCapability(FEATURE_TYPE_MEMORY, static_cast<int32_t>(MEMORY_INFO_TS_LIMITED), &value);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rt api rtGetRtCapability failed, ret = " << rt_ret;
  }
  auto memory_type = (value == static_cast<int64_t>(RT_CAPABILITY_SUPPORT)) ? RT_MEMORY_TS : RT_MEMORY_HBM;
  rt_ret = rtMalloc(&op_debug_buffer_addr_, kOpDebugHostMemSize, memory_type, 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rt api rtMalloc failed, ret = " << rt_ret;
  }

  rt_ret = rtMalloc(&op_debug_dump_args_, kOpDebugDevMemSize, RT_MEMORY_HBM, 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtMalloc failed, ret = " << rt_ret;
  }

  rt_ret =
    aclrtMemcpy(op_debug_dump_args_, sizeof(void *), &op_debug_buffer_addr_, sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call aclrtMemcpy failed, ret = " << rt_ret;
  }

  rt_ret = rtDebugRegister(model_handle_(), op_debug_mode, op_debug_buffer_addr_, &debug_stream_id_, &debug_task_id_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtDebugRegister failed, ret = " << rt_ret;
  }

  MS_LOG(INFO) << "[DataDump] Distribute op debug task, task id = " << debug_task_id_
               << ", stream id = " << debug_stream_id_;
}

void DataDumper::OpDebugUnregister() {
  uint32_t op_debug_mode = DumpJsonParser::GetInstance().op_debug_mode();
  if (op_debug_mode == kNoOverflow) {
    MS_LOG(INFO) << "[DataDump] Op debug mode is no overflow, no need to unregister.";
    return;
  }

  MS_LOG(INFO) << "[DataDump] Start.";
  rtError_t rt_ret = rtDebugUnRegister(model_handle_());
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "[DataDump] Call rtDebugUnRegister failed, ret = " << rt_ret;
    return;
  }
}
#endif

#ifndef ENABLE_SECURITY
void DataDumper::DumpKernelOutput(const CNodePtr &kernel, void *args, NotNull<aicpu::dump::Task *> task) {
  if (!DumpJsonParser::GetInstance().OutputNeedDump()) {
    MS_LOG(INFO) << "Skip dump output";
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  if (HasAbstractMonad(kernel)) {
    MS_LOG(WARNING) << "Skip Monad node output:" << kernel->fullname_with_scope();
    return;
  }
  MS_LOG(INFO) << "[DataDump] DumpKernelOutput start. Kernel:" << kernel->fullname_with_scope();
  auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);
  auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
  uint64_t offset = sizeof(void *) * input_size;
  for (size_t i = 0; i < output_size; ++i) {
    auto data_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
    auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel, i);
    auto output_origin_shape = common::AnfAlgo::GetOutputInferShape(kernel, i);

    aicpu::dump::Output output;
    output.set_data_type(static_cast<int>(GeTypesConvert::GetGeDataType(data_type)));
    output.set_format(static_cast<int>(GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    SetDumpShape(output_shape, NOT_NULL(output.mutable_shape()));
    SetDumpShape(output_origin_shape, NOT_NULL(output.mutable_origin_shape()));

    output.set_original_output_format(
      static_cast<int>(GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    output.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args)) + offset);
    // device address data size
    auto address = AnfAlgo::GetOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(address);
    output.set_size(address->GetSize());
    MS_LOG(INFO) << "[DataDump] output " << i << " address size:" << output.size();
    MS_EXCEPTION_IF_NULL(task->mutable_output());
    task->mutable_output()->Add(std::move(output));
    offset = SizetAddWithOverflowCheck(offset, sizeof(void *));
  }
}

void DataDumper::DumpKernelInput(const CNodePtr &kernel, void *args, NotNull<aicpu::dump::Task *> task) {
  if (!DumpJsonParser::GetInstance().InputNeedDump()) {
    MS_LOG(INFO) << "Skip dump input";
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  if (common::AnfAlgo::IsNodeInputContainMonad(kernel)) {
    MS_LOG(WARNING) << "Skip Monad node:" << kernel->fullname_with_scope();
    return;
  }
  MS_LOG(INFO) << "[DataDump] DumpKernelInput start. Kernel:" << kernel->fullname_with_scope();
  auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);
  uint64_t offset = 0;
  for (size_t i = 0; i < input_size; ++i) {
    auto real_index = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, i);
    if (common::AnfAlgo::IsNoneInput(kernel, real_index)) {
      continue;
    }
    aicpu::dump::Input input;
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, real_index);
    auto input_node = input_node_with_index.first;
    auto input_index = input_node_with_index.second;
    std::string output_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(input_node, input_index);
    if (output_type == kTypeUnknown) {
      MS_LOG(WARNING) << "[DataDump] It is not suggested to use a lonely weight parameter as the output of graph";
      output_type = common::AnfAlgo::GetOutputInferDataType(input_node, input_index);
    }
    auto output_shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    auto output_origin_shape = common::AnfAlgo::GetOutputInferShape(input_node, input_index);

    input.set_data_type(static_cast<int>(GeTypesConvert::GetGeDataType(output_type)));
    input.set_format(static_cast<int>(GeTypesConvert::GetGeFormat(output_format, output_shape.size())));
    SetDumpShape(output_shape, NOT_NULL(input.mutable_shape()));
    SetDumpShape(output_origin_shape, NOT_NULL(input.mutable_origin_shape()));

    input.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args)) + offset);
    // device  address data size
    auto address = AnfAlgo::GetPrevNodeOutputAddr(kernel, real_index);
    MS_EXCEPTION_IF_NULL(address);
    input.set_size(address->GetSize());
    MS_LOG(INFO) << "[DataDump] input " << i << " address size:" << input.size();
    MS_EXCEPTION_IF_NULL(task->mutable_input());
    task->mutable_input()->Add(std::move(input));
    offset = SizetAddWithOverflowCheck(offset, sizeof(void *));
  }
}
#endif

std::string DataDumper::StripUniqueId(const std::string node_name) {
  size_t last_underscore = node_name.find_last_of('_');
  std::string stripped_node_name;
  if (last_underscore == string::npos) {
    MS_LOG(ERROR) << "Could not strip unique ID from " << node_name;
    stripped_node_name = node_name;
  } else {
    stripped_node_name = node_name.substr(0, last_underscore);
  }
  return stripped_node_name;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
