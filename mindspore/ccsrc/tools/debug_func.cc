/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tools/debug_func.h"
#include <mutex>
#include <vector>
#include <memory>
#include <string>
#include <set>
#include <algorithm>
#include "tools/data_dump/dump_json_parser.h"
#include "include/utils/anfalgo.h"
#include "tools/data_dump/cpu_e2e_dump.h"
#include "tools/data_dump/data_dump.h"
#include "tools/silent_detect/checksum/checksum.h"
#include "tools/silent_detect/silent_detector.h"
#include "utils/log_adapter.h"
#ifdef ENABLE_DEBUGGER
#include "tools/data_dump/debugger/debugger.h"
#include "tools/data_dump/device_statistic/mem_manager.h"
#endif

namespace mindspore {

namespace tools {
namespace {
std::mutex debug_mutex;
}  // namespace

void DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs, const std::vector<AnfNodePtr> &origin_parameters_order,
                      std::vector<DeviceContext *> device_contexts) {
  std::lock_guard<std::mutex> locker(debug_mutex);

  MS_VLOG(VL_DUMP) << "Debug on step begin.";
  if (DumpJsonParser::GetInstance().e2e_dump_enabled() && !graphs.empty()) {
    // First graph is the dataset graph when dataset_sink_mode = True
    auto graph = graphs[0];
    bool is_dataset_graph = graph->IsDatasetGraph();
    uint32_t cur_step = DumpJsonParser::GetInstance().cur_dump_iter();
    if (cur_step == 1 && DumpJsonParser::GetInstance().GetDatasetSink()) {
      uint32_t init_step = 0;
      DumpJsonParser::GetInstance().UpdateDumpIter(init_step);
      MS_VLOG(VL_DUMP) << "In dataset sink mode, reset step to init_step: " << init_step;
    }
    DumpJsonParser::GetInstance().SetDatasetSink(is_dataset_graph);
  }
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr && debugger->DebuggerBackendEnabled()) {
    debugger->PreExecuteGraphDebugger(graphs, origin_parameters_order);
  }
#endif
  if (DumpJsonParser::GetInstance().e2e_dump_enabled()) {
    DumpJsonParser::GetInstance().ClearGraph();
    if (graphs.size() != device_contexts.size()) {
      MS_LOG(EXCEPTION) << "Graph num:" + std::to_string(graphs.size()) +
                             " is not equal to device context size:" + std::to_string(device_contexts.size()) +
                             " for debug actor.";
    }
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(graphs[i]);
      MS_EXCEPTION_IF_NULL(device_contexts[i]);
      if (device_contexts[i]->GetDeviceType() == device::DeviceType::kCPU) {
        DumpJsonParser::GetInstance().SaveGraph(graphs[i].get());
      }
    }
  }
}

void DebugPostLaunch(const AnfNodePtr &node, const std::vector<kernel::KernelTensorPtr> &input_kernel_tensors,
                     const std::vector<kernel::KernelTensorPtr> &output_kernel_tensors,
                     const DeviceContext *device_context) {
  std::lock_guard<std::mutex> locker(debug_mutex);

  std::vector<kernel::KernelTensor *> raw_input_kernel_tensors;
  raw_input_kernel_tensors.resize(input_kernel_tensors.size());
  std::vector<kernel::KernelTensor *> raw_output_kernel_tensors;
  raw_output_kernel_tensors.resize(output_kernel_tensors.size());

  std::transform(input_kernel_tensors.begin(), input_kernel_tensors.end(), raw_input_kernel_tensors.begin(),
                 [](const kernel::KernelTensorPtr &ptr) { return ptr.get(); });
  std::transform(output_kernel_tensors.begin(), output_kernel_tensors.end(), raw_output_kernel_tensors.begin(),
                 [](const kernel::KernelTensorPtr &ptr) { return ptr.get(); });

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);

  if (!node->isa<CNode>()) {
    return;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_VLOG(VL_DUMP) << "kernel by kernel debug for node: " << cnode->fullname_with_scope() << ", device type is "
                   << device_context->GetDeviceType();
  if (device_context->GetDeviceType() == device::DeviceType::kAscend) {
    checksum::AscendCheckSum(cnode, raw_input_kernel_tensors, raw_output_kernel_tensors, device_context);
  }
  datadump::DataDump(cnode, raw_input_kernel_tensors, raw_output_kernel_tensors, device_context);
}

void DebugOnStepEnd(int total_running_count, std::vector<const DeviceContext *> device_contexts) {
  std::lock_guard<std::mutex> locker(debug_mutex);

  MS_VLOG(VL_DUMP) << "Debug on step end. total_running_count is: " << total_running_count
                   << "; total user_dump_step is: " << DumpJsonParser::GetInstance().cur_dump_iter();
  std::set<const DeviceContext *> sync_stream_device_contexts;
  for (auto &device_context : device_contexts) {
    MS_EXCEPTION_IF_NULL(device_context);
    if ((sync_stream_device_contexts.count(device_context) == 0) &&
        (!device_context->device_res_manager_->SyncAllStreams())) {
      MS_LOG(ERROR) << "Sync stream failed:" + device_context->device_context_key().ToString();
    }
    (void)sync_stream_device_contexts.insert(device_context);
  }

  if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
    CPUE2eDump::DumpParametersData();
    CPUE2eDump::DumpConstantsData();
  }

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    debugger->Debugger::PostExecuteGraphDebugger();
  }
  DumpJsonParser::GetInstance().UpdateDumpIter(total_running_count);
  MS_VLOG(VL_DUMP) << "UpdateDumpIter: " << total_running_count;
#endif
}

void DebugFinalize() {
  DumpJsonParser::GetInstance().PrintUnusedKernel();
#ifdef ENABLE_DEBUGGER
  datadump::DumpMemManager::GetInstance().ClearCache();
#endif
  silentdetect::SilentDetector::Stop();
}
}  // namespace tools
}  // namespace mindspore
