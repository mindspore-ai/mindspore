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

#include "tools/data_dump/data_dump.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tools/data_dump/cpu_e2e_dump.h"
#include "tools/data_dump/e2e_dump.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#ifdef ENABLE_DEBUGGER
#include "tools/data_dump/debugger/debugger.h"
#include "tools/data_dump/debugger/debugger_utils.h"
#include "tools/data_dump/device_statistic/mem_manager.h"
#endif
#include "include/utils/anfalgo.h"
#include "include/utils/callback.h"
#include "include/utils/common.h"
#include "tools/data_dump/dump_json_parser.h"
#include "tools/silent_detect/checksum/checksum.h"
#include "tools/tensor_dump/tensordump_utils.h"

namespace {
static const char kTensorDumpFlag[] = "td_flag";
static const char kNameSeparator[] = "|";
}  // namespace
namespace mindspore::datadump {
#ifdef ENABLE_DEBUGGER
void AscendDataDump(const CNodePtr &cnode, const std::vector<kernel::KernelTensor *> &input_kernel_tensors,
                    const std::vector<kernel::KernelTensor *> &output_kernel_tensors,
                    const device::DeviceContext *device_context) {
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    debugger->InsertExecutedGraph(kernel_graph);
    debugger->SetAscendKernelByKernelFlag(true);
    auto &dump_json_parser = DumpJsonParser::GetInstance();
    bool e2e_dump_enabled = dump_json_parser.e2e_dump_enabled();
    uint32_t op_debug_mode = dump_json_parser.op_debug_mode();
    bool abnormal_dump = false;
    bool sync_ok = true;
    bool read_data = false;
    if (!e2e_dump_enabled) {
      return;
    }
    if (op_debug_mode == DumpJsonParser::DUMP_LITE_EXCEPTION) {
      abnormal_dump = true;
      sync_ok = device_context->device_res_manager_->SyncAllStreams();
      if (!sync_ok) {
        MS_LOG(ERROR) << "Sync stream error! The node input will be dumped";
      }
    } else if (op_debug_mode == DumpJsonParser::DUMP_BOTH_OVERFLOW && dump_json_parser.DumpEnabledForIter()) {
      read_data = true;
    } else {
      read_data = CheckReadData(cnode);
    }
    if ((read_data && e2e_dump_enabled) || !sync_ok) {
      string scope_name;
      if (common::AnfAlgo::HasNodeAttr(kTensorDumpFlag, cnode)) {
        scope_name = cnode->fullname_with_scope();
        auto first_input = cnode->input(1);
        MS_EXCEPTION_IF_NULL(first_input);
        auto abs = first_input->abstract();
        MS_EXCEPTION_IF_NULL(abs);
        auto input_value_track = abs->GetValueTrack();
        MS_EXCEPTION_IF_NULL(input_value_track);
        auto input_value = dyn_cast_ptr<StringImm>(input_value_track);
        MS_EXCEPTION_IF_NULL(input_value);
        string input_str = input_value->value();
        string new_scope_name = input_str + kNameSeparator + scope_name;
        cnode->set_fullname_with_scope(new_scope_name);
      }
      if (dump_json_parser.e2e_sync_dump_enabled()) {
        ReadDataAndDump(cnode, input_kernel_tensors, output_kernel_tensors, device_context, abnormal_dump);
      } else {
        DumpDataViaCallback(cnode, input_kernel_tensors, output_kernel_tensors, device_context);
      }
      if (common::AnfAlgo::HasNodeAttr(kTensorDumpFlag, cnode)) {
        cnode->set_fullname_with_scope(scope_name);
      }

      if (!sync_ok) {
        MS_LOG(EXCEPTION) << "Sync stream error!";
      }
    }
    datadump::DumpMemManager::GetInstance().Reset();
  }
}

void GPUDataDump(const CNodePtr &cnode, std::vector<kernel::KernelTensor *> input_kernel_tensors,
                 std::vector<kernel::KernelTensor *> output_kernel_tensors,
                 const device::DeviceContext *device_context) {
  if (DumpJsonParser::GetInstance().op_debug_mode() == DumpJsonParser::DUMP_LITE_EXCEPTION) {
    MS_LOG(WARNING) << "Abnormal dump is not supported on GPU backend.";
    return;
  }
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
    debugger->InsertExecutedGraph(kernel_graph);
    bool read_data = CheckReadData(cnode);
    if (read_data) {
      ReadDataAndDump(cnode, input_kernel_tensors, output_kernel_tensors, device_context);
    }
  }
}

#endif

void CPUDataDump(const CNodePtr &cnode) {
  if (DumpJsonParser::GetInstance().op_debug_mode() == DumpJsonParser::DUMP_LITE_EXCEPTION) {
    MS_LOG(WARNING) << "Abnormal dump is not supported on CPU backend.";
    return;
  }
  if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
    auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    CPUE2eDump::DumpCNodeData(cnode, kernel_graph->graph_id());
    CPUE2eDump::DumpRunIter(kernel_graph);
  }
}

void DataDump(const CNodePtr &cnode, const std::vector<kernel::KernelTensor *> &input_kernel_tensors,
              const std::vector<kernel::KernelTensor *> &output_kernel_tensors,
              const device::DeviceContext *device_context) {
  if (device_context->GetDeviceType() == device::DeviceType::kAscend) {
#ifdef ENABLE_DEBUGGER
    datadump::AscendDataDump(cnode, input_kernel_tensors, output_kernel_tensors, device_context);
#endif
  } else if (device_context->GetDeviceType() == device::DeviceType::kCPU) {
    datadump::CPUDataDump(cnode);
  } else if (device_context->GetDeviceType() == device::DeviceType::kGPU) {
#ifdef ENABLE_DEBUGGER
    datadump::GPUDataDump(cnode, input_kernel_tensors, output_kernel_tensors, device_context);
#endif
  }
}

}  // namespace mindspore::datadump
