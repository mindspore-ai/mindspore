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

#include "plugin/device/ascend/kernel/pyboost/customize/stress_detect.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

int LaunchAclnnWithNoInput(const std::string &aclnn_name, const device::DeviceContext *device_context,
                           size_t real_stream_id) {
  runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);
  auto stream_ptr = device_context->device_res_manager_->GetStream(real_stream_id);
  auto workspace_api_name = aclnn_name + "GetWorkspaceSize";
  const auto get_workspace_size_func_ptr = transform::GetOpApiFunc(workspace_api_name.c_str());
  if (get_workspace_size_func_ptr == nullptr) {
    MS_LOG(EXCEPTION) << workspace_api_name << " not in " << transform::GetOpApiLibName() << ", please check!";
  }
  uint64_t workspace_size = 0;
  transform::aclOpExecutor *executor = nullptr;
  using OpApiFunc = int (*)(size_t *, transform::aclOpExecutor **);
  auto func = reinterpret_cast<OpApiFunc>(get_workspace_size_func_ptr);
  func(&workspace_size, &executor);
  void *workspace_addr = nullptr;
  if (workspace_size != 0) {
    auto workspace_device_address = runtime::DeviceAddressUtils::CreateWorkspaceAddressWithoutKernelTensor(
      device_context, real_stream_id, workspace_size, true);
    if (workspace_device_address->GetMutablePtr() == nullptr) {
      MS_LOG(WARNING) << " Can't allocate workspace memory size: " << workspace_size << " for " << aclnn_name;
      return 0;
    }
    workspace_addr = workspace_device_address->GetMutablePtr();
  }
  const auto op_api_func = transform::GetOpApiFunc(aclnn_name.c_str());
  if (op_api_func == nullptr) {
    MS_LOG(EXCEPTION) << aclnn_name << " not in " << transform::GetOpApiLibName() << ", please check!";
  }
  auto run_api_func = reinterpret_cast<transform::RunApiFunc>(op_api_func);
  auto api_ret = run_api_func(workspace_addr, workspace_size, executor, stream_ptr);
  return api_ret;
}

int StressDetectKernel(const device::DeviceContext *device_context, size_t stream_id) {
  constexpr int success = 0;
  auto stage_1_result = LaunchAclnnWithNoInput("aclnnStressDetect", device_context, stream_id);
  auto stage_2_result = LaunchAclnnWithNoInput("aclnnStressDetectWithPressure", device_context, stream_id);
  if (stage_1_result != success) {
    return stage_1_result;
  }
  if (stage_2_result != success) {
    return stage_2_result;
  }
  return success;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
