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

#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include <vector>
#include <string>
#include <map>
#include "common/util/error_manager/error_manager.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/ms_context.h"
#include "runtime/dev.h"
#include "acl/error_codes/rt_error_codes.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::map<uint32_t, std::string> error_msg = {
  {ACL_RT_SUCCESS, "success"},
  {ACL_ERROR_RT_PARAM_INVALID, "param invalid"},
  {ACL_ERROR_RT_INVALID_DEVICEID, "invalid device id"},
  {ACL_ERROR_RT_CONTEXT_NULL, "current context null"},
  {ACL_ERROR_RT_STREAM_CONTEXT, "stream not in current context"},
  {ACL_ERROR_RT_MODEL_CONTEXT, "model not in current context"},
  {ACL_ERROR_RT_STREAM_MODEL, "stream not in model"},
  {ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID, "event timestamp invalid"},
  {ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL, " event timestamp reversal"},
  {ACL_ERROR_RT_ADDR_UNALIGNED, "memory address unaligned"},
  {ACL_ERROR_RT_FILE_OPEN, "open file failed"},
  {ACL_ERROR_RT_FILE_WRITE, "write file failed"},
  {ACL_ERROR_RT_STREAM_SUBSCRIBE, "error subscribe stream"},
  {ACL_ERROR_RT_THREAD_SUBSCRIBE, "error subscribe thread"},
  {ACL_ERROR_RT_GROUP_NOT_SET, "group not set"},
  {ACL_ERROR_RT_GROUP_NOT_CREATE, "group not create"},
  {ACL_ERROR_RT_STREAM_NO_CB_REG, "callback not register to stream"},
  {ACL_ERROR_RT_INVALID_MEMORY_TYPE, "invalid memory type"},
  {ACL_ERROR_RT_INVALID_HANDLE, "invalid handle"},
  {ACL_ERROR_RT_INVALID_MALLOC_TYPE, "invalid malloc type"},
  {ACL_ERROR_RT_FEATURE_NOT_SUPPORT, "feature not support"},
  {ACL_ERROR_RT_MEMORY_ALLOCATION, "memory allocation error"},
  {ACL_ERROR_RT_MEMORY_FREE, "memory free error"},
  {ACL_ERROR_RT_AICORE_OVER_FLOW, "aicore over flow"},
  {ACL_ERROR_RT_NO_DEVICE, "no device"},
  {ACL_ERROR_RT_RESOURCE_ALLOC_FAIL, "resource alloc fail"},
  {ACL_ERROR_RT_NO_PERMISSION, "no permission"},
  {ACL_ERROR_RT_NO_EVENT_RESOURCE, "no event resource"},
  {ACL_ERROR_RT_NO_STREAM_RESOURCE, "no stream resource"},
  {ACL_ERROR_RT_NO_NOTIFY_RESOURCE, "no notify resource"},
  {ACL_ERROR_RT_NO_MODEL_RESOURCE, "no model resource"},
  {ACL_ERROR_RT_INTERNAL_ERROR, "runtime internal error"},
  {ACL_ERROR_RT_TS_ERROR, "ts internal error"},
  {ACL_ERROR_RT_STREAM_TASK_FULL, "task full in stream"},
  {ACL_ERROR_RT_STREAM_TASK_EMPTY, " task empty in stream"},
  {ACL_ERROR_RT_STREAM_NOT_COMPLETE, "stream not complete"},
  {ACL_ERROR_RT_END_OF_SEQUENCE, "end of sequence"},
  {ACL_ERROR_RT_EVENT_NOT_COMPLETE, "event not complete"},
  {ACL_ERROR_RT_CONTEXT_RELEASE_ERROR, "context release error"},
  {ACL_ERROR_RT_SOC_VERSION, "soc version error"},
  {ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT, "task type not support"},
  {ACL_ERROR_RT_LOST_HEARTBEAT, "ts lost heartbeat"},
  {ACL_ERROR_RT_MODEL_EXECUTE, " model execute failed"},
  {ACL_ERROR_RT_REPORT_TIMEOUT, "report timeout"},
  {ACL_ERROR_RT_SYS_DMA, "sys dma error"},
  {ACL_ERROR_RT_AICORE_TIMEOUT, "aicore timeout"},
  {ACL_ERROR_RT_AICORE_EXCEPTION, "aicore exception"},
  {ACL_ERROR_RT_AICORE_TRAP_EXCEPTION, " aicore trap exception"},
  {ACL_ERROR_RT_AICPU_TIMEOUT, " aicpu timeout"},
  {ACL_ERROR_RT_AICPU_EXCEPTION, "aicpu exception"},
  {ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR, " aicpu datadump response error"},
  {ACL_ERROR_RT_AICPU_MODEL_RSP_ERR, "aicpu model operate response error"},
  {ACL_ERROR_RT_PROFILING_ERROR, "profiling error"},
  {ACL_ERROR_RT_IPC_ERROR, "ipc error"},
  {ACL_ERROR_RT_MODEL_ABORT_NORMAL, "model abort normal"},
  {ACL_ERROR_RT_KERNEL_UNREGISTERING, "kernel unregistering"},
  {ACL_ERROR_RT_RINGBUFFER_NOT_INIT, "ringbuffer not init"},
  {ACL_ERROR_RT_RINGBUFFER_NO_DATA, "ringbuffer no data"},
  {ACL_ERROR_RT_KERNEL_LOOKUP, "kernel lookup error"},
  {ACL_ERROR_RT_KERNEL_DUPLICATE, "kernel register duplicate"},
  {ACL_ERROR_RT_DEBUG_REGISTER_FAIL, "debug register failed"},
  {ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL, "debug unregister failed"},
  {ACL_ERROR_RT_LABEL_CONTEXT, "label not in current context"},
  {ACL_ERROR_RT_PROGRAM_USE_OUT, "program register num use out"},
  {ACL_ERROR_RT_DEV_SETUP_ERROR, "device setup error"},
  {ACL_ERROR_RT_DRV_INTERNAL_ERROR, "drv internal error"},
};

constexpr auto kUnknowErrorString = "Unknown error occurred";
constexpr auto kSOC_VERSION = "SOC_VERSION";
}  // namespace

std::string GetErrorMessage(bool add_title) {
  const string &error_message = ErrorManager::GetInstance().GetErrorMessage();
  if (error_message.empty() || error_message.find(kUnknowErrorString) != string::npos) {
    return "";
  }
  if (add_title) {
    return "#umsg#Ascend Error Message:#umsg#" + error_message +
           "\n(Please search \"Ascend Error Message\" at https://www.mindspore.cn for error code description)";
  }
  return error_message;
}

void SetErrorManagerContext() { ErrorManager::GetInstance().GenWorkStreamIdDefault(); }

std::string GetWarningMessage() {
  const string &warning_message = ErrorManager::GetInstance().GetWarningMessage();
  if (!warning_message.empty()) {
    return warning_message;
  }
  return "";
}

bool IsGraphMode() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
}

bool IsDynamicShapeGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  return std::any_of(node_list.begin(), node_list.end(),
                     [](const AnfNodePtr &node) { return common::AnfAlgo::IsDynamicShape(node); });
}

std::string GetSocVersion() {
  // Get default soc version.
  static std::string version{};
  if (version.empty()) {
    const int kSocVersionLen = 50;
    char soc_version[kSocVersionLen] = {0};
    auto ret = rtGetSocVersion(soc_version, kSocVersionLen);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "GetSocVersion failed.";
    }
    // Get soc version from env value.
    const char *soc_version_env = nullptr;
    std::string str_soc_version_env = common::GetEnv(kSOC_VERSION);
    if (!str_soc_version_env.empty()) {
      soc_version_env = common::SafeCStr(str_soc_version_env);
    }
    if (soc_version_env != nullptr) {
      if (std::strcmp(soc_version, soc_version_env) != 0) {
        MS_LOG(DEBUG) << "Detected the env SOC_VERSION, so the SocVersion will be changed to " << str_soc_version_env
                      << ".";
        ret = rtSetSocVersion(soc_version_env);
        if (ret != RT_ERROR_NONE) {
          MS_LOG(EXCEPTION) << "SetSocVersion failed, errorno: " << ret;
        }
        version = soc_version_env;
        return soc_version_env;
      }
    }
    version = soc_version;
  }
  return version;
}

void AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  for (auto output : outputs) {
    if (!output->isa<CNode>() || !AnfUtils::IsRealKernel(output)) {
      continue;
    }

    if (!common::AnfAlgo::IsNopNode(output)) {
      continue;
    }

    if (!common::AnfAlgo::IsNeedSkipNopOpAddr(output)) {
      continue;
    }

    size_t input_num = common::AnfAlgo::GetInputTensorNum(output);
    if (input_num != 1) {
      MS_LOG(WARNING) << "The input number of nop node :" << output->fullname_with_scope() << " is " << input_num
                      << ", not equal 1";
      continue;
    }

    auto real_input_index = AnfAlgo::GetInputGraphIdxByKernelIdx(output, 0);
    auto pre_node_out_device_address = AnfAlgo::GetPrevNodeOutputAddr(output, real_input_index);
    MS_EXCEPTION_IF_NULL(pre_node_out_device_address);
    auto ptr = pre_node_out_device_address->GetPtr();
    auto size = pre_node_out_device_address->GetSize();
    std::string output_format = AnfAlgo::GetOutputFormat(output, 0);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(output, 0);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
      const_cast<void *>(ptr), size, output_format, output_type, trans::GetRuntimePaddingShape(output, 0));
    // If graph has the flag kFlagEnableZeroCopyInGraph, it means the graph should run in graph mode, the device
    // address of graph output should not be persisted, and its output addr will be replaced after RunGraph.
    // As we fetch the output device address of a nopnode, we can only get the input device address of it, so we
    // have to prevent the ptr persist flag of the device address here.
    if (!graph->has_flag(kFlagEnableZeroCopyInGraph)) {
      device_address->set_is_ptr_persisted(true);
    }
    device_address->set_host_shape(trans::GetRuntimePaddingShape(output, 0));
    AnfAlgo::SetOutputAddr(device_address, 0, output.get());
    common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(false), output);
    MS_LOG(INFO) << "Assign device address to output nop node " << output->fullname_with_scope();
  }
}

std::string GetErrorMsg(uint32_t rt_error_code) {
  auto find_iter = error_msg.find(rt_error_code);
  if (find_iter == error_msg.end()) {
    return "Return error code unknown, ret code: " + std::to_string(rt_error_code);
  }
  return find_iter->second;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
