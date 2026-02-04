/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pybind_api/tools/stress_detect_py.h"
#include <utility>
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_context.h"

#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "mindspore/ccsrc/pybind_api/tools/tools_api.h"

namespace mindspore {
namespace {
device::DeviceContext *GetDeviceCtx() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name), MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_ctx);

  device_ctx->Initialize();
  return device_ctx;
}
}  // namespace

int StressDetect(const std::string &detect_type) {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx);
  runtime::Pipeline::Get().WaitAll();
  auto &controller =
    device::DeviceContextManager::GetInstance().GetMultiStreamController(device_ctx->device_context_key().device_type_);
  controller->Refresh();
  (void)controller->SyncAllStreams();
  MS_EXCEPTION_IF_NULL(device_ctx->GetKernelExecutor());
  return device_ctx->GetKernelExecutor()->StressDetect(detect_type);
}

void RegStress(py::module *m) {
  (void)m->def("stress_detect", &mindspore::StressDetect, "Detect stress", py::arg("detect_type") = "aic");
}
}  // namespace mindspore
