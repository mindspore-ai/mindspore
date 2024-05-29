/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/debug_actor.h"
#include <vector>
#include <memory>
#include <string>
#include "runtime/graph_scheduler/actor/debug_aware_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/cpu_e2e_dump.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#include "utils/ms_context.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#include "debug/debugger/debugger_utils.h"
#endif
#include "debug/data_dump/data_dumper.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "include/backend/debug/profiler/profiling.h"
#include "ops/nn_op_name.h"

namespace mindspore {
namespace runtime {
void DebugActor::ACLDump(uint32_t device_id, const std::vector<KernelGraphPtr> &graphs, bool is_kbyk) {
  std::string env_enable_str = common::GetEnv("MS_ACL_DUMP_CFG_PATH");
  std::string dump_enable_str = common::GetEnv("MINDSPORE_DUMP_CONFIG");
  auto step_count_num = 0;
  step_count_num = step_count;
  if (step_count == 1 && is_dataset_sink == 1) {
    step_count_num = 0;
  }
  if (!graphs.empty()) {
    auto graph = graphs[0];
    is_dataset_sink = graph->IsDatasetGraph();
  }
  if (DumpJsonParser::GetInstance().async_dump_enabled() &&
      ((DumpJsonParser::GetInstance().IsDumpIter(step_count_num) && is_kbyk) ||
       (env_enable_str == dump_enable_str && !is_kbyk))) {
    bool is_init = false;
    if ((env_enable_str == dump_enable_str) && !(DumpJsonParser::GetInstance().IsDumpIter(step_count_num))) {
      is_init = true;
    } else {
      std::string dump_path = DumpJsonParser::GetInstance().path();
      std::string dump_path_step = dump_path + "/" + std::to_string(step_count_num);
      auto real_path = FileUtils::CreateNotExistDirs(dump_path_step, false);
      if (!real_path.has_value()) {
        MS_LOG(WARNING) << "Fail to create acl dump dir " << real_path.value();
        return;
      }
    }
    dump_flag = true;
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      registered_dumper->Initialize();
      registered_dumper->EnableDump(device_id, step_count_num, is_init);
    }
  }
}
/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Load and read data for the given node if needed. Dump the node if dump is enabled and free the loaded
 * memory after the dump (for GPU and ascend kernel-by-kernel).
 */
void DebugActor::Debug(const AnfNodePtr &node, const KernelLaunchAddr *launch_info,
                       const std::vector<KernelTensor *> &op_output_kernel_tensors, const DeviceContext *device_context,
                       OpContext<DeviceTensor> *const op_context, const AID *) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);

  if (!node->isa<CNode>()) {
    return;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "kernel by kernel debug for node: " << cnode->fullname_with_scope() << ".";
  if (device_context->GetDeviceType() == device::DeviceType::kAscend) {
#ifdef ENABLE_DEBUGGER
    auto debugger = Debugger::GetInstance();
    if (debugger != nullptr) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      debugger->InsertExecutedGraph(kernel_graph);
      debugger->SetAscendKernelByKernelFlag(true);
      bool read_data = CheckReadData(cnode);
      if (read_data && DumpJsonParser::GetInstance().e2e_dump_enabled()) {
        ReadDataAndDump(cnode, launch_info, exec_order_, device_context);
      }
    }
    exec_order_ += 1;
#endif
  } else if (device_context->GetDeviceType() == device::DeviceType::kCPU) {
#ifndef ENABLE_SECURITY
    if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      MS_EXCEPTION_IF_NULL(kernel_graph);
      CPUE2eDump::DumpCNodeData(cnode, kernel_graph->graph_id());
      CPUE2eDump::DumpRunIter(kernel_graph);
    }
#endif
  } else if (device_context->GetDeviceType() == device::DeviceType::kGPU) {
#ifdef ENABLE_DEBUGGER
    auto debugger = Debugger::GetInstance();
    if (debugger != nullptr) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      debugger->InsertExecutedGraph(kernel_graph);
      std::string kernel_name = cnode->fullname_with_scope();
      debugger->SetCurNode(kernel_name);
      bool read_data = CheckReadData(cnode);
      if (read_data) {
        ReadDataAndDump(cnode, launch_info, exec_order_, device_context);
      }
    }
    exec_order_ += 1;
#endif
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Checks dataset_sink_mode and generates the related error if any exist and calls
 * PreExecuteGraphDebugger.
 */
void DebugActor::DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                                  const std::vector<AnfNodePtr> &origin_parameters_order,
                                  std::vector<DeviceContext *> device_contexts,
                                  OpContext<DeviceTensor> *const op_context, const AID *) {
  MS_LOG(INFO) << "Debug on step begin.";
  auto context = MsContext::GetInstance();
  auto is_kbyk = context->IsKByKExecutorMode();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  device_ctx_ = device_contexts[0];
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if ((profiler == nullptr || !profiler->IsInitialized()) &&
      device_ctx_->GetDeviceType() == device::DeviceType::kAscend) {
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    if (common::GetEnv("MS_ACL_DUMP_CFG_PATH") == common::GetEnv("MINDSPORE_DUMP_CONFIG")) {
      ACLDump(device_id, graphs, is_kbyk);
    }
  }
#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().e2e_dump_enabled() && !graphs.empty()) {
    // First graph is the dataset graph when dataset_sink_mode = True
    auto graph = graphs[0];
    bool is_dataset_sink = graph->IsDatasetGraph();
    uint32_t cur_step = DumpJsonParser::GetInstance().cur_dump_iter();
    if (cur_step == 1 && DumpJsonParser::GetInstance().GetDatasetSink()) {
      uint32_t init_step = 0;
      DumpJsonParser::GetInstance().UpdateDumpIter(init_step);
      MS_LOG(INFO) << "In dataset sink mode, reset step to init_step: " << init_step;
    }
    DumpJsonParser::GetInstance().SetDatasetSink(is_dataset_sink);
  }
#endif
  if (backend == "ge") {
    return;
  }
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);
#ifdef ENABLE_DEBUGGER
  if (!graphs.empty()) {
    // First graph is the dataset graph when dataset_sink_mode = True
    auto graph = graphs[0];
    std::string error_info = CheckDatasetSinkMode(graph);
    if (!error_info.empty()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
    }
  }
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr && debugger->DebuggerBackendEnabled()) {
    debugger->PreExecuteGraphDebugger(graphs, origin_parameters_order);
  }
#endif
#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().e2e_dump_enabled()) {
    DumpJsonParser::GetInstance().ClearGraph();
    if (graphs.size() != device_contexts.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), "Graph num:" + std::to_string(graphs.size()) +
                                                         " is not equal to device context size:" +
                                                         std::to_string(device_contexts.size()) + " for debug actor.");
    }
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(graphs[i]);
      MS_EXCEPTION_IF_NULL(device_contexts[i]);
      if (device_contexts[i]->GetDeviceType() == device::DeviceType::kCPU) {
        DumpJsonParser::GetInstance().SaveGraph(graphs[i].get());
      }
    }
  }
#endif
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 * Description: Dump parameters and constants and update dump iter for CPU. Call PostExecuteGraph Debugger for GPU and
 * Ascend and update step number of online debugger GPU.
 */
void DebugActor::DebugOnStepEnd(OpContext<DeviceTensor> *const op_context, const AID *, int total_running_count_) {
  MS_LOG(INFO) << "Debug on step end. total_running_count is: " << total_running_count_;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  step_count = total_running_count_;
  if (dump_flag == true) {
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      device_ctx_->device_res_manager_->SyncAllStreams();
      registered_dumper->Finalize();
    }
    dump_flag = false;
  }
  auto is_kbk = context->IsKByKExecutorMode();
  device_ctx_->device_res_manager_->SyncAllStreams();
  if (backend == "ge" && !is_kbk) {
    return;
  }
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);

#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
    CPUE2eDump::DumpParametersData();
    CPUE2eDump::DumpConstantsData();
  }
#endif

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    // Reset exec_order for the next step
    exec_order_ = 0;
    debugger->Debugger::PostExecuteGraphDebugger();
    debugger->Debugger::UpdateStepNumGPU();
  }
#ifndef ENABLE_SECURITY
  DumpJsonParser::GetInstance().UpdateDumpIter(step_count);
  MS_LOG(INFO) << "UpdateDumpIter: " << step_count;
#endif
#endif
}

bool DebugActor::CheckFinite(const DeviceContext *device_context, const std::vector<KernelTensor *> &inputs) {
  if (inputs.empty()) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  // 1. Get AllFinite kernel mod.
  const auto &kernel_mod_iter = finite_kernel_mods_.find(device_context);
  kernel::KernelModPtr finite_kernel_mod = nullptr;
  if (kernel_mod_iter == finite_kernel_mods_.end()) {
    const auto &new_finite_kernel_mod = device_context->GetKernelExecutor(false)->CreateKernelMod(kAllFiniteOpName);
    MS_EXCEPTION_IF_NULL(new_finite_kernel_mod);
    finite_kernel_mods_.emplace(device_context, new_finite_kernel_mod);
    finite_kernel_mod = new_finite_kernel_mod;
  } else {
    finite_kernel_mod = kernel_mod_iter->second;
  }
  MS_EXCEPTION_IF_NULL(finite_kernel_mod);

  // 2. Get output kernel tensor for AllFinite kernel.
  MS_EXCEPTION_IF_NULL(inputs[0]);
  const auto &stream_id = inputs[0]->stream_id();
  auto &stream_id_to_output_device_address = finite_output_device_addresses_[device_context];
  if (stream_id_to_output_device_address.find(stream_id) == stream_id_to_output_device_address.end()) {
    auto finite_output_addr = device_context->device_res_manager_->AllocateMemory(1, stream_id);
    MS_EXCEPTION_IF_NULL(finite_output_addr);

    ShapeVector shape_vec = {1};
    auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
      finite_output_addr, 1, Format::DEFAULT_FORMAT, kNumberTypeBool, shape_vec,
      device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
    kernel_tensor->set_stream_id(stream_id);
    kernel_tensor->SetType(std::make_shared<TensorType>(kBool));
    kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(shape_vec));
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    MS_EXCEPTION_IF_NULL(device_address);
    stream_id_to_output_device_address.emplace(stream_id, device_address);
  }
  auto &output_device_address = stream_id_to_output_device_address[stream_id];
  MS_EXCEPTION_IF_NULL(output_device_address);
  const auto &output_kernel_tensor = output_device_address->kernel_tensor();
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);

  void *stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  bool ret = finite_kernel_mod->Launch(inputs, {}, {output_kernel_tensor.get()}, stream_ptr);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Launch AllFinite kernel failed.";
  }
  return output_kernel_tensor->GetValueWithCheck<bool>();
}

void DebugActor::Finalize() {
  for (const auto &item : finite_output_device_addresses_) {
    auto &stream_id_to_output_device_address_map = item.second;
    auto *device_context = item.first;
    for (const auto &device_address_item : stream_id_to_output_device_address_map) {
      const auto &device_address = device_address_item.second;
      if (device_address && device_context) {
        device_context->device_res_manager_->FreeMemory(device_address->GetMutablePtr());
      }
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
