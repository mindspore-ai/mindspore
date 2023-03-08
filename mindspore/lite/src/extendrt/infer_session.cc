/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/infer_session.h"

#include "extendrt/session/single_op_session.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"
#include "backend/graph_compiler/graph_partition.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "extendrt/utils/kernel_graph_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/delegate/plugin/tensorrt_executor_plugin.h"
#include "extendrt/delegate/plugin/litert_executor_plugin.h"
#include "extendrt/delegate/plugin/ascend_ge_executor_plugin.h"

namespace mindspore {
std::shared_ptr<InferSession> InferSession::CreateSession(const std::shared_ptr<Context> &context,
                                                          const ConfigInfos &config_info) {
  HandleContext(context);
  auto session_type = SelectSession(context);
  MS_LOG(DEBUG) << "Session type " << static_cast<int64_t>(session_type);
  return SessionRegistry::GetInstance().GetSession(session_type, context, config_info);
}

void InferSession::HandleContext(const std::shared_ptr<Context> &context) {
  if (!context) {
    return;
  }
  constexpr auto default_gpu_provider = "tensorrt";
  constexpr auto default_cpu_provider = "litert";
  constexpr auto default_npu_provider = "ge";
  auto device_infos = context->MutableDeviceInfo();
  for (auto &device_info : device_infos) {
    if (!device_info) {
      continue;
    }
    if (device_info->GetDeviceType() == kGPU) {
      auto gpu_device = device_info->Cast<GPUDeviceInfo>();
      if (!gpu_device) {
        continue;
      }
      auto provider = gpu_device->GetProvider();
      if (provider.empty() || provider == default_gpu_provider) {
        if (!lite::TensorRTExecutorPlugin::GetInstance().Register()) {
          MS_LOG_WARNING << "Failed to register TensorRT plugin";
          return;
        }
        gpu_device->SetProvider(default_gpu_provider);
      }
      continue;
    }
    if (device_info->GetDeviceType() == kAscend) {
      auto ascend_device = device_info->Cast<AscendDeviceInfo>();
      if (!ascend_device) {
        continue;
      }
      auto provider = ascend_device->GetProvider();
      if (provider == default_npu_provider) {
        if (!lite::AscendGeExecutorPlugin::GetInstance().Register()) {
          MS_LOG_WARNING << "Failed to register AscendGe plugin";
          return;
        }
      }
    }
    if (device_info->GetDeviceType() == kCPU) {
      auto cpu_device = device_info->Cast<CPUDeviceInfo>();
      if (!cpu_device) {
        continue;
      }
      auto provider = cpu_device->GetProvider();
      if (provider.empty() || provider == default_cpu_provider) {
        if (!infer::LiteRTExecutorPlugin::GetInstance().Register()) {
          MS_LOG_WARNING << "Failed to register LiteRT plugin";
          return;
        }
        cpu_device->SetProvider(default_cpu_provider);
      }
      continue;
    }
    if (device_info->GetDeviceType() == kAllDevice) {
      // Auto Device: MSLite will detect available device and run graph/sub-graph on suitable device by its scheduler
      continue;
    }
  }
}

SessionType InferSession::SelectSession(const std::shared_ptr<Context> &context) {
  if (context != nullptr) {
    auto &device_contexts = context->MutableDeviceInfo();
    for (auto device_context : device_contexts) {
      MS_EXCEPTION_IF_NULL(device_context);
      if (device_context->GetDeviceType() == kAscend) {
        if (device_context->GetProvider() == "ge") {
          return kDelegateSession;
        }
        return kSingleOpSession;
      }
      if (device_context->GetDeviceType() == kGPU || device_context->GetDeviceType() == kCPU) {
        return kDelegateSession;
      }
      if (device_context->GetDeviceType() == kAllDevice) {
        // Default Session support auto device context
        return kDefaultSession;
      }
    }
  }
  return kDefaultSession;
}
}  // namespace mindspore
