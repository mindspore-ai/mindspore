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
#include "pybind_api/runtime/res_limit_py.h"
#include <utility>
#include <string>
#include <memory>
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "utils/ms_context.h"
#include "pybind_api/runtime/utils_py.h"
#include "pybind_api/runtime/runtime_api.h"
#include "include/pynative/utils/pynative_utils.h"
#include "utils/stream_guard.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace hal {
py::tuple GetDeviceLimit(int32_t device_id) {
  runtime::Pipeline::Get().WaitForward();
  uint32_t cube_num;
  uint32_t vector_num;
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name), static_cast<uint32_t>(device_id)});
  device_context->Initialize();
  device_context->device_res_manager_->GetDeviceLimit(device_id, &cube_num, &vector_num);
  return py::make_tuple(cube_num, vector_num);
}

void SetDeviceLimit(int32_t device_id, int32_t cube_num, int32_t vector_num) {
  runtime::Pipeline::Get().WaitForward();
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name), static_cast<uint32_t>(device_id)});
  device_context->Initialize();
  device_context->device_res_manager_->SetDeviceLimit(device_id, cube_num, vector_num);
}

void ResetStreamLimit(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(stream);
  runtime::Pipeline::Get().WaitForward();
  MS_LOG(DEBUG) << "stream_id:" << stream->stream_id();
  stream->device_ctx()->device_res_manager_->ResetStreamLimit(stream->stream_id());
}

py::tuple GetStreamLimit(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(stream);
  runtime::Pipeline::Get().WaitForward();
  MS_LOG(DEBUG) << "stream_id:" << stream->stream_id();
  uint32_t cube_num;
  uint32_t vector_num;
  stream->device_ctx()->device_res_manager_->GetStreamLimit(stream->stream_id(), &cube_num, &vector_num);
  return py::make_tuple(cube_num, vector_num);
}

void SetStreamLimit(const StreamPyPtr &stream, int32_t cube_num, int32_t vector_num) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_LOG(DEBUG) << "stream_id:" << stream->stream_id();
  DispatchSetStreamLimitTask(stream, cube_num, vector_num);
}

void DispatchSetStreamLimitTask(const StreamPyPtr &stream, int32_t cube_num, int32_t vector_num) {
  // Wait set stream limit async.
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>([stream, cube_num, vector_num]() {
    auto wait_fn = [stream, cube_num, vector_num]() {
      auto stream_id = stream->stream_id();
      MS_LOG(DEBUG) << "WaitEvent wait stream id:" << stream_id << ", cube_num:" << cube_num
                    << ", vectore_num:" << vector_num;
      auto device_ctx = stream->device_ctx();
      MS_EXCEPTION_IF_NULL(device_ctx);
      runtime::OpExecutor::DispatchLaunchTask([stream_id, cube_num, vector_num, device_ctx]() {
        device_ctx->device_res_manager_->SetStreamLimit(stream_id, cube_num, vector_num);
      });
    };
    if (!runtime::OpExecutor::NeedSync()) {
      runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
        std::make_shared<runtime::PassthroughNoWaitDeviceTask>(wait_fn));
    } else {
      wait_fn();
    }
  }));
}

void RegResLimit(py::module *m) {
  (void)m->def("get_device_limit", &mindspore::hal::GetDeviceLimit, "Get device limit");
  (void)m->def("set_device_limit", &mindspore::hal::SetDeviceLimit, "Set device limit");
  (void)m->def("get_stream_limit", &mindspore::hal::GetStreamLimit, "Get stream limit");
  (void)m->def("set_stream_limit", &mindspore::hal::SetStreamLimit, "Set stream limit");
  (void)m->def("reset_stream_limit", &mindspore::hal::ResetStreamLimit, "Reset stream limit");
}
}  // namespace hal
}  // namespace mindspore
