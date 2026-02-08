/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/stress_detect/stress_detect.h"
#include <string>
#include <thread>
#include <future>
#include <memory>
#include <utility>
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "mindspore/core/include/utils/device_manager_conf.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

void StressDetectTask::Run() {
  auto ret = run_func_(device_id_, workspace_addr_, workspace_size_);
  p_.set_value(ret);
}

void AmlAicoreDetectTask::Run() {
  auto ret = run_func_(device_id_, attr_.get());
  p_.set_value(ret);
}

int LaunchAclnnWithNoInput(const std::string &aclnn_name, const device::DeviceContext *device_context) {
  runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);
  uint64_t workspace_size = 10;
  constexpr uint64_t kSize = 1024;
  workspace_size = workspace_size * kSize * kSize * kSize;
  void *workspace_addr = nullptr;
  auto workspace_device_address = runtime::DeviceAddressUtils::CreateWorkspaceAddressWithoutKernelTensor(
    device_context, device_context->device_res_manager_->DefaultStream(), workspace_size, true);
  if (workspace_device_address->GetMutablePtr() == nullptr) {
    MS_LOG(WARNING) << " Can't allocate workspace memory size: " << workspace_size << " for " << aclnn_name;
    return kDetectFailed;
  }
  workspace_addr = workspace_device_address->GetMutablePtr();
  std::promise<int> p;
  std::future<int> f = p.get_future();
  std::shared_ptr<runtime::AsyncTask> task;

  if (aclnn_name == "AmlAicoreDetectOnline") {
    auto ascend_path = mindspore::device::ascend::GetAscendPath();
    auto lib_path = ascend_path + GetLibAscendMLName();
    void *lib_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (lib_handle == nullptr) {
      MS_LOG(EXCEPTION) << lib_path << " was not found. Exiting stress detect";
    }
    const auto *op_api_func = dlsym(lib_handle, aclnn_name.c_str());
    if (op_api_func == nullptr) {
      MS_LOG(EXCEPTION) << aclnn_name << " not in " << GetLibAscendMLName() << ", please check!";
    }
    auto run_api_func = reinterpret_cast<int (*)(int32_t, const AmlAicoreDetectAttr *)>(op_api_func);

    auto aml_attr = std::make_shared<AmlAicoreDetectAttr>();
    aml_attr->mode = AML_DETECT_RUN_MODE_ONLINE;
    aml_attr->workspaceSize = workspace_size;
    aml_attr->workspace = workspace_addr;

    task = std::make_shared<AmlAicoreDetectTask>(
      std::move(run_api_func), device_context->device_context_key().device_id_, aml_attr, std::move(p));
    auto aml_task = std::dynamic_pointer_cast<AmlAicoreDetectTask>(task);
    MS_LOG(DEBUG) << "aml_task created with device_id: " << aml_task->device_id()
                  << ", attr.runmode: " << aml_task->attr()->mode
                  << ", attr.workspaceSize: " << aml_task->attr()->workspaceSize
                  << ", attr.workspace: " << aml_task->attr()->workspace;
  } else {
    const auto op_api_func = device::ascend::GetOpApiFunc(aclnn_name.c_str());
    if (op_api_func == nullptr) {
      MS_LOG(EXCEPTION) << aclnn_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!";
    }
    auto run_api_func = reinterpret_cast<int (*)(int32_t, void *, uint64_t)>(op_api_func);
    task = std::make_shared<StressDetectTask>(std::move(run_api_func), device_context->device_context_key().device_id_,
                                              workspace_addr, workspace_size, std::move(p));
  }
  runtime::Pipeline::Get().stress_detect()->Push(task);
  runtime::Pipeline::Get().stress_detect()->Wait();
  int api_ret = f.get();
  return api_ret;
}

int LaunchHccsTask(const device::DeviceContext *device_context, void *group_comm) {
  runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostLaunchAclnn, kNameAmlP2PDetectOnline, false);
  uint64_t workspace_size = 10;
  constexpr uint64_t kSize = 1024;
  workspace_size = workspace_size * kSize * kSize * kSize;
  void *workspace_addr = nullptr;
  auto ascend_path = mindspore::device::ascend::GetAscendPath();
  auto lib_path = ascend_path + GetLibAscendMLName();
  void *lib_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (lib_handle == nullptr) {
    MS_LOG(EXCEPTION) << lib_path << " was not found. Exiting HCCS stress detect task.";
  }
  const auto *amlp2p_func = dlsym(lib_handle, kNameAmlP2PDetectOnline);
  if ((amlp2p_func == nullptr) || (group_comm == nullptr)) {
    MS_LOG(WARNING) << kNameAmlP2PDetectOnline << " not found in CANN or group comm not found, skipping P2P test.";
    return kDetectFailed;
  } else {
    workspace_addr = device::ascend::AscendMemAdapter::GetInstance()->MallocAlign32FromRts(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(WARNING) << " Can't allocate workspace memory size: " << workspace_size << " for HCCS stress detect task";
      return kDetectFailed;
    }
    auto p2p_api_func = reinterpret_cast<int (*)(int32_t, void *, const AmlP2PDetectAttr *)>(amlp2p_func);
    AmlP2PDetectAttr p2p_attr;
    p2p_attr.workspaceSize = workspace_size;
    p2p_attr.workspace = workspace_addr;
    int ret = p2p_api_func(device_context->device_context_key().device_id_, group_comm, &p2p_attr);

    MS_LOG(INFO) << "P2P detection executed - device_id: " << device_context->device_context_key().device_id_
                 << ", workspaceSize: " << p2p_attr.workspaceSize << ", workspace: " << p2p_attr.workspace
                 << ", comm: " << group_comm;

    device::ascend::AscendMemAdapter::GetInstance()->FreeAlign32ToRts(workspace_addr);
    return ret;
  }
}

int StressDetectKernel(const std::string &detect_type) {
  auto ascend_path = mindspore::device::ascend::GetAscendPath();
  auto lib_path = ascend_path + GetLibAscendMLName();
  int ret;
  auto device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name), device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  void *lib_handle = dlopen(lib_path.c_str(), RTLD_LAZY);

  if (detect_type == "aic") {
    if (lib_handle) {
      void *func_ptr = dlsym(lib_handle, kNameAmlAicoreDetectOnline);
      if (func_ptr) {
        MS_LOG(INFO) << "Using new API AmlAicoreDetectOnline from " << lib_path;
        ret = LaunchAclnnWithNoInput("AmlAicoreDetectOnline", device_context);
      } else {
        MS_LOG(INFO) << "AmlAicoreDetectOnline not found in " << lib_path << ". Using the StressDetect api instead.";
        ret = LaunchAclnnWithNoInput("StressDetect", device_context);
      }
      dlclose(lib_handle);
    } else {
      MS_LOG(INFO) << lib_path << " not found. Using the StressDetect api instead.";
      ret = LaunchAclnnWithNoInput("StressDetect", device_context);
    }
    MS_LOG(WARNING) << "Aicore stress detect end; return code is [" << ret << "]";
    constexpr int clear_device_state_fail = 574007;
    if (ret == clear_device_state_fail) {
      MS_LOG(EXCEPTION) << "Stress detect: clear device state fail!";
    }
  } else {
    if (lib_handle) {
      void *group_comm = device::ascend::AscendCollectiveCommLib::GetInstance().GetHcomByGroup(detect_type);
      ret = LaunchHccsTask(device_context, group_comm);
      MS_LOG(WARNING) << "HCCS stress detect end; return code is [" << ret << "]";
    } else {
      MS_LOG(WARNING) << lib_path << " not found. Skipping HCCS stress detect task.";
      return kDetectFailed;
    }
  }
  constexpr int STRESS_BIT_FAIL = 574006;
  constexpr int STRESS_LOW_BIT_FAIL = 574008;
  constexpr int STRESS_HIGH_BIT_FAIL = 574009;
  if (ret == 0) {
    MS_LOG(INFO) << "Stress detect successful";
    return kDetectSucceeded;
  } else if (ret == STRESS_BIT_FAIL || ret == STRESS_LOW_BIT_FAIL || ret == STRESS_HIGH_BIT_FAIL) {
    MS_LOG(WARNING) << "Stress detect failed with hardware fault. This could lead to accuracy issues in training";
    return kDetectFailedWithHardwareFailure;
  } else {
    MS_LOG(WARNING) << "Stress detect failed because some or all test cases were not implemented.";
    return kDetectFailed;
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
