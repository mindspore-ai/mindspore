/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "runtime/device/ascend/dump/data_dumper.h"

#include <map>
#include <memory>
#include <string>
#include <algorithm>
#include "utility"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "runtime/rt_model.h"
#include "runtime/device/ascend/ge_types_convert.h"
#include "proto/op_mapping_info.pb.h"
#include "utils/ms_context.h"
#include "debug/data_dump/dump_json_parser.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif

static constexpr uint32_t kAicpuLoadFlag = 1;
static constexpr uint32_t kAicpuUnloadFlag = 0;
static constexpr uint32_t kTupleTaskId = 0;
static constexpr uint32_t kTupleStreamId = 1;
static constexpr uint32_t kTupleArgs = 2;
static constexpr uint32_t kCurrentStepTensorIndex = 0;
static constexpr uint32_t kCurrentEpochTensorIndex = 2;
static constexpr uint32_t kStepsPerEpochTensorIndex = 3;
static constexpr uint64_t kOpDebugShape = 2048;
static constexpr uint64_t kOpDebugHostMemSize = 2048;
static constexpr uint64_t kOpDebugDevMemSize = sizeof(void *);
static constexpr uint8_t kNoOverflow = 0;
static constexpr uint8_t kAiCoreOverflow = (0x1 << 0);
static constexpr uint8_t kAtomicOverflow = (0x1 << 1);
static constexpr uint8_t kAllOverflow = (kAiCoreOverflow | kAtomicOverflow);
static const std::map<uint32_t, std::string> kOverflowModeStr = {{kNoOverflow, "NoOverflow"},
                                                                 {kAiCoreOverflow, "AiCoreOverflow"},
                                                                 {kAtomicOverflow, "AtomicOverflow"},
                                                                 {kAllOverflow, "AllOverflow"}};
constexpr const char *kNodeNameOpDebug = "Node_OpDebug";
constexpr const char *kOpTypeOpDebug = "Opdebug";

namespace mindspore {
namespace device {
namespace ascend {
DataDumper::~DataDumper() {
  ReleaseDevMem(&dev_load_mem_);
  ReleaseDevMem(&dev_unload_mem_);
  ReleaseDevMem(&op_debug_buffer_addr_);
  ReleaseDevMem(&op_debug_dump_args_);
}

void DataDumper::GetNeedDumpKernelList(NotNull<std::map<std::string, CNodePtr> *> kernel_map) const {
  for (const auto &kernel : kernel_graph_->execution_order()) {
    if (AnfAlgo::GetKernelType(kernel) == HCCL_KERNEL &&
        DumpJsonParser::GetInstance().NeedDump(kernel->fullname_with_scope())) {
      auto input_size = AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_size; ++i) {
        auto input_with_index = AnfAlgo::GetPrevNodeOutput(kernel, i);
        auto input = input_with_index.first;
        if (input->isa<CNode>()) {
          MS_LOG(INFO) << "[AsyncDump] Match Hccl Node:" << kernel->fullname_with_scope()
                       << " Input:" << input->fullname_with_scope();
          kernel_map->try_emplace(input->fullname_with_scope(), input->cast<CNodePtr>());
        }
      }
    } else if (KernelNeedDump(kernel)) {
      MS_LOG(INFO) << "[AsyncDump] Match Node:" << kernel->fullname_with_scope();
      kernel_map->try_emplace(kernel->fullname_with_scope(), kernel);
    }
  }
}

void DataDumper::LoadDumpInfo() {
  MS_LOG(INFO) << "[DataDump] LoadDumpInfo start";
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  aicpu::dump::OpMappingInfo dump_info;
  SetOpDebugMappingInfo(NOT_NULL(&dump_info));
  SetOpMappingInfo(NOT_NULL(&dump_info));

  auto kernels = kernel_graph_->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (!KernelNeedDump(kernel)) {
      continue;
    }
    MS_LOG(INFO) << "[DataDump] LoadDumpInfo kernel:" << kernel->fullname_with_scope();
    dump_kernel_names_.emplace_back(kernel->fullname_with_scope());
    DumpJsonParser::GetInstance().MatchKernel(kernel->fullname_with_scope());

    aicpu::dump::Task task;
    ConstructDumpTask(NOT_NULL(kernel), NOT_NULL(&task));
    MS_EXCEPTION_IF_NULL(dump_info.mutable_task());
    dump_info.mutable_task()->Add(std::move(task));
  }
  RtLoadDumpData(dump_info, &dev_load_mem_);
  load_flag_ = true;
  // graph id may changed in Unload
  graph_id_ = kernel_graph_->graph_id();
#ifdef ENABLE_DEBUGGER
  auto debugger = mindspore::Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->DebuggerBackendEnabled()) {
    std::map<std::pair<uint32_t, uint32_t>, std::string> stream_task_to_opname;
    // extract stream id, task id and opname from runtime_info_map for overflow detection
    std::transform(runtime_info_map_.begin(), runtime_info_map_.end(),
                   std::inserter(stream_task_to_opname, stream_task_to_opname.end()),
                   [](const std::pair<std::string, std::shared_ptr<RuntimeInfo>> &p)
                     -> std::pair<std::pair<uint32_t, uint32_t>, std::string> {
                     return {{std::get<1>(*p.second), std::get<0>(*p.second)}, p.first};
                   });
    debugger->SetStreamTaskToOpnameMap(stream_task_to_opname);
  }
#endif
  MS_LOG(INFO) << "[DataDump] LoadDumpInfo end";
}

void DataDumper::SetOpMappingInfo(NotNull<aicpu::dump::OpMappingInfo *> dump_info) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  auto dump_path = DumpJsonParser::GetInstance().path();
  if (dump_path.empty()) {
    MS_LOG(EXCEPTION) << "Dump path invalid";
  }
  auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  dump_info->set_dump_path("/" + dump_path + "/device_" + std::to_string(device_id) + "/");
  MS_LOG(INFO) << "[DataDump] dump_path:" << dump_path;

  dump_info->set_model_name(DumpJsonParser::GetInstance().net_name() + "_graph_" +
                            std::to_string(kernel_graph_->graph_id()));
  dump_info->set_dump_step(std::to_string(DumpJsonParser::GetInstance().iteration()));
  dump_info->set_model_id(kernel_graph_->graph_id());
  dump_info->set_flag(kAicpuLoadFlag);

  const auto &input_ctrl_tensors = kernel_graph_->input_ctrl_tensors();
  if (input_ctrl_tensors == nullptr || input_ctrl_tensors->size() < 3) {
    MS_LOG(INFO) << "[DataDump] Not data sink mode, input_ctrl_tensor";
    return;
  }
  const auto &current_step_tensor = input_ctrl_tensors->at(kCurrentStepTensorIndex);
  const auto &currnet_epoch_tensor = input_ctrl_tensors->at(kCurrentEpochTensorIndex);
  const auto &steps_per_epoch_tensor = input_ctrl_tensors->at(kStepsPerEpochTensorIndex);

  MS_EXCEPTION_IF_NULL(current_step_tensor);
  MS_EXCEPTION_IF_NULL(currnet_epoch_tensor);
  MS_EXCEPTION_IF_NULL(steps_per_epoch_tensor);
  MS_EXCEPTION_IF_NULL(current_step_tensor->device_address());
  MS_EXCEPTION_IF_NULL(currnet_epoch_tensor->device_address());
  MS_EXCEPTION_IF_NULL(steps_per_epoch_tensor->device_address());

  void *current_step = current_step_tensor->device_address()->GetMutablePtr();
  void *current_epoch = currnet_epoch_tensor->device_address()->GetMutablePtr();
  void *steps_per_epoch = steps_per_epoch_tensor->device_address()->GetMutablePtr();

  if (current_epoch != nullptr && current_step != nullptr && steps_per_epoch != nullptr) {
    dump_info->set_step_id_addr(reinterpret_cast<uint64_t>(current_epoch));
    dump_info->set_loop_cond_addr(reinterpret_cast<uint64_t>(current_step));
    dump_info->set_iterations_per_loop_addr(reinterpret_cast<uint64_t>(steps_per_epoch));
  } else {
    MS_LOG(INFO) << "Invalid ctrl tensor device address";
  }
}

bool DataDumper::KernelNeedDump(const CNodePtr &kernel) const {
  if (AnfAlgo::GetKernelType(kernel) != TBE_KERNEL && AnfAlgo::GetKernelType(kernel) != AICPU_KERNEL &&
      AnfAlgo::GetKernelType(kernel) != AKG_KERNEL) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  // dump all kernel if mode is set 0 in data_dump.json
  return DumpJsonParser::GetInstance().NeedDump(kernel->fullname_with_scope());
}

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
    auto iter = runtime_info_map_.find(kernel_name);
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

void DataDumper::ReleaseDevMem(void **ptr) const {
  if (ptr == nullptr) {
    return;
  }
  if (*ptr != nullptr) {
    rtError_t rt_error = rtFree(*ptr);
    if (rt_error != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "[DataDump] Call rtFree failed, ret:" << rt_error;
    }
    *ptr = nullptr;
  }
}

void DataDumper::ConstructDumpTask(NotNull<const CNodePtr &> kernel, NotNull<aicpu::dump::Task *> dump_task) const {
  dump_task->set_end_graph(false);
  auto iter = runtime_info_map_.find(kernel->fullname_with_scope());
  if (iter == runtime_info_map_.end()) {
    MS_LOG(EXCEPTION) << "[DataDump] kernel name not found in runtime_info_map";
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  auto task_id = std::get<kTupleTaskId>(*iter->second);
  auto stream_id = std::get<kTupleStreamId>(*iter->second);
  auto args = std::get<kTupleArgs>(*iter->second);
  MS_LOG(INFO) << "[DataDump] Get runtime info task_id:" << task_id << " stream_id:" << stream_id;

  dump_task->set_task_id(task_id);
  dump_task->set_stream_id(stream_id);
  MS_EXCEPTION_IF_NULL(dump_task->mutable_op());
  dump_task->mutable_op()->set_op_name(kernel->fullname_with_scope());
  dump_task->mutable_op()->set_op_type(AnfAlgo::GetCNodeName(kernel.get()));

  DumpKernelOutput(kernel, args, dump_task);
  DumpKernelInput(kernel, args, dump_task);
}

void DataDumper::SetOpDebugMappingInfo(const NotNull<aicpu::dump::OpMappingInfo *> dump_info) const {
  MS_LOG(INFO) << "[DataDump] Add op debug info to OpMappingInfo, task id = " << debug_task_id_
               << ", stream id = " << debug_stream_id_;
  aicpu::dump::Task task;
  task.set_end_graph(false);
  task.set_task_id(debug_task_id_);
  task.set_stream_id(debug_stream_id_);
  task.mutable_op()->set_op_name(kNodeNameOpDebug);
  task.mutable_op()->set_op_type(kOpTypeOpDebug);

  aicpu::dump::Output output;
  output.set_data_type(ge::proto::DataType::DT_UINT8);
  output.set_format(ge::Format::FORMAT_ND);

  output.mutable_shape()->add_dim(kOpDebugShape);

  output.set_original_name(kNodeNameOpDebug);
  output.set_original_output_index(0);
  output.set_original_output_format(ge::Format::FORMAT_ND);
  output.set_original_output_data_type(ge::proto::DataType::DT_UINT8);
  // due to lhisi virtual addr bug, cannot use args now
  output.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(op_debug_dump_args_)));
  output.set_size(kOpDebugHostMemSize);

  task.mutable_output()->Add(std::move(output));
  dump_info->mutable_task()->Add(std::move(task));
}

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

  rtError_t rt_ret = rtMalloc(&op_debug_buffer_addr_, kOpDebugHostMemSize, RT_MEMORY_DDR);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtMalloc failed, ret = " << rt_ret;
  }

  rt_ret = rtMalloc(&op_debug_dump_args_, kOpDebugDevMemSize, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtMalloc failed, ret = " << rt_ret;
  }

  rt_ret =
    rtMemcpy(op_debug_dump_args_, sizeof(void *), &op_debug_buffer_addr_, sizeof(void *), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtMemcpy failed, ret = " << rt_ret;
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
    MS_LOG(EXCEPTION) << "[DataDump] Call rtDebugUnRegister failed, ret = " << rt_ret;
  }
}

void DataDumper::RtLoadDumpData(const aicpu::dump::OpMappingInfo &dump_info, void **ptr) {
  std::string proto_str;
  size_t proto_size = dump_info.ByteSizeLong();
  bool ret = dump_info.SerializeToString(&proto_str);
  if (!ret || proto_size == 0) {
    MS_LOG(EXCEPTION) << "[DataDump] Protobuf SerializeToString failed, proto size %zu.";
  }

  if (ptr == nullptr) {
    MS_LOG(ERROR) << "[DataDump] rtMalloc failed, ptr is nullptr";
    return;
  }

  rtError_t rt_ret = rtMalloc(ptr, proto_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtMalloc failed";
  }
  rt_ret = rtMemcpy(*ptr, proto_size, proto_str.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtMemcpy failed";
  }

  MS_LOG(INFO) << "[DataDump] rtDatadumpInfoLoad start";
  rt_ret = rtDatadumpInfoLoad(*ptr, proto_size);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DataDump] Call rtDatadumpInfoLoad failed";
  }
}

void SetDumpShape(const std::vector<size_t> &ms_shape, NotNull<aicpu::dump::Shape *> dump_shape) {
  for (auto &dim : ms_shape) {
    dump_shape->add_dim(dim);
  }
}

void DataDumper::DumpKernelOutput(const CNodePtr &kernel, void *args, NotNull<aicpu::dump::Task *> task) {
  if (!DumpJsonParser::GetInstance().OutputNeedDump()) {
    MS_LOG(INFO) << "Skip dump output";
    return;
  }
  MS_LOG(INFO) << "[DataDump] DumpKernelOutput start. Kernel:" << kernel->fullname_with_scope();
  auto input_size = AnfAlgo::GetInputTensorNum(kernel);
  auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
  uint64_t offset = sizeof(void *) * input_size;
  for (size_t i = 0; i < output_size; ++i) {
    auto data_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
    auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel, i);
    auto output_origin_shape = AnfAlgo::GetOutputInferShape(kernel, i);

    aicpu::dump::Output output;
    output.set_data_type(GeTypesConvert::GetGeDataType(data_type));
    output.set_format(GeTypesConvert::GetGeFormat(output_format, output_shape.size()));
    SetDumpShape(output_shape, NOT_NULL(output.mutable_shape()));
    SetDumpShape(output_origin_shape, NOT_NULL(output.mutable_origin_shape()));

    output.set_original_output_format(GeTypesConvert::GetGeFormat(output_format, output_shape.size()));
    output.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args)) + offset);
    // device address data size
    auto address = AnfAlgo::GetOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(address);
    output.set_size(address->GetSize());
    MS_LOG(INFO) << "[DataDump] output " << i << " address size:" << output.size();
    MS_EXCEPTION_IF_NULL(task->mutable_output());
    task->mutable_output()->Add(std::move(output));
    offset += sizeof(void *);
  }
}

void DataDumper::DumpKernelInput(const CNodePtr &kernel, void *args, NotNull<aicpu::dump::Task *> task) {
  if (!DumpJsonParser::GetInstance().InputNeedDump()) {
    MS_LOG(INFO) << "Skip dump input";
    return;
  }
  MS_LOG(INFO) << "[DataDump] DumpKernelInput start. Kernel:" << kernel->fullname_with_scope();
  auto input_size = AnfAlgo::GetInputTensorNum(kernel);
  uint64_t offset = 0;
  for (size_t i = 0; i < input_size; ++i) {
    aicpu::dump::Input input;
    auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(kernel, i);
    auto input_node = input_node_with_index.first;
    auto input_index = input_node_with_index.second;
    std::string output_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(input_node, input_index);
    if (output_type == kTypeUnknown) {
      MS_LOG(WARNING) << "[DataDump] It is not suggested to use a lonely weight parameter as the output of graph";
      output_type = AnfAlgo::GetOutputInferDataType(input_node, input_index);
    }
    auto output_shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    auto output_origin_shape = AnfAlgo::GetOutputInferShape(input_node, input_index);

    input.set_data_type(GeTypesConvert::GetGeDataType(output_type));
    input.set_format(GeTypesConvert::GetGeFormat(output_format, output_shape.size()));
    SetDumpShape(output_shape, NOT_NULL(input.mutable_shape()));
    SetDumpShape(output_origin_shape, NOT_NULL(input.mutable_origin_shape()));

    input.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args)) + offset);
    // device  address data size
    auto address = AnfAlgo::GetPrevNodeOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(address);
    input.set_size(address->GetSize());
    MS_LOG(INFO) << "[DataDump] input " << i << " address size:" << input.size();
    MS_EXCEPTION_IF_NULL(task->mutable_input());
    task->mutable_input()->Add(std::move(input));
    offset += sizeof(void *);
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
