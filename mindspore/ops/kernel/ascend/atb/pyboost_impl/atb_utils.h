/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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
#ifndef MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_UTILS_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_UTILS_H_

#include <vector>
#include <string>
#include <memory>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "ir/tensor.h"
#include "kernel/ascend/atb/pyboost_impl/atb_runner_base.h"

namespace mindspore::kernel::pyboost {
template <typename ParamType>
bool LaunchAtb(const std::string &atb_name, ParamType *param, const device::DeviceContext *device_context,
               uint32_t stream_id, const std::vector<tensor::TensorPtr> &inputs,
               const std::vector<tensor::TensorPtr> &outputs) {
  static auto simu = common::IsCompileSimulation();
  if (simu) {
    return true;
  }
  runtime::ProfilerRecorder atb_profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostLaunchAtb,
                                         atb_name, false);
  static bool is_load_ = false;
  static std::shared_ptr<ATBRunnerBase> atb_runner = nullptr;
  if (!is_load_) {
    is_load_ = true;
    // try load so file
    std::string cur_so_path;
    (void)plugin_loader::PluginLoader::GetPluginPath(&cur_so_path);
    auto target_so_path = cur_so_path + "/ascend/" + "libmindspore_pyboost_atb_kernels.so";
    auto ret = dlopen(target_so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    auto err_msg = dlerror();
    if (!ret) {
      MS_LOG(WARNING) << "Load ATB so " << target_so_path << " failed, error: " << err_msg
                      << " , you can enable ATB by install the nnal package and source the set_env.sh in nnal.";
      return false;
    }
    atb_runner = Factory<ATBRunnerBase>::Instance().Create(atb_name + "ATBRunner");
  }
  if (atb_runner == nullptr) {
    return false;
  }
  atb_runner->Init(atb_name, param);
  atb_runner->GetWorkSpaceInfo(device_context, stream_id, inputs, outputs);
  atb_runner->Run(stream_id, device_context);
  return true;
}
}  // namespace mindspore::kernel::pyboost
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_UTILS_H_
