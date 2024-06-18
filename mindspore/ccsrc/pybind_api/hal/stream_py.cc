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
#include "pybind_api/hal/stream_py.h"
#include <utility>
#include "runtime/pynative/op_executor.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/common/pybind_api/api_register.h"
#include "runtime/device/multi_stream_controller.h"

namespace mindspore {
namespace hal {
namespace {
DeviceContext *GetDeviceCtx() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_name, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_ctx);

  device_ctx->Initialize();
  return device_ctx;
}
}  // namespace

StreamPy::StreamPy(int priority) {
  device_ctx_ = GetDeviceCtx();
  device_ctx_->device_res_manager_->CreateStreamWithPriority(&stream_id_, priority);
  MS_LOG(DEBUG) << "stream_id:" << stream_id_ << ", priority:" << priority;
  device_ctx_->device_res_manager_->set_single_op_multi_stream_enable(true);
}

StreamPy::StreamPy(int priority, int stream_id) {
  device_ctx_ = GetDeviceCtx();
  stream_id_ = IntToSize(stream_id);
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  if (device_ctx_->device_res_manager_->GetStream(stream_id) == nullptr) {
    MS_EXCEPTION(ValueError) << "stream_id:" << stream_id << " is not exist";
  }
}

StreamPy::~StreamPy() { device_ctx_ = nullptr; }

bool StreamPy::Query() {
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  runtime::Pipeline::Get().WaitForward();
  return device_ctx_->device_res_manager_->QueryStream(stream_id_);
}

void StreamPy::Synchronize() {
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  runtime::Pipeline::Get().WaitForward();
  device::MultiStreamController::GetInstance()->Refresh(device_ctx_);
  (void)device::MultiStreamController::GetInstance()->SyncStream(device_ctx_, stream_id_);
}

std::string StreamPy::ToStringRepr() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(device_ctx_);
  buf << "Stream(device_name=" << device_ctx_->device_context_key().device_name_
      << ", device_id:" << std::to_string(device_ctx_->device_context_key().device_id_)
      << ", stream id:" << std::to_string(stream_id_) << ")";
  return buf.str();
}

void *StreamPy::stream() const {
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  return device_ctx_->device_res_manager_->GetStream(stream_id_);
}

bool StreamPy::StreamEqual(const std::shared_ptr<StreamPy> other_stream) {
  MS_EXCEPTION_IF_NULL(other_stream);

  MS_LOG(DEBUG) << "stream info:" << ToStringRepr() << " other_stream info:" << other_stream->ToStringRepr();
  return (stream_id_ == other_stream->stream_id()) &&
         (device_ctx()->device_context_key().device_name_ ==
          other_stream->device_ctx()->device_context_key().device_name_) &&
         (device_ctx()->device_context_key().device_id_ == other_stream->device_ctx()->device_context_key().device_id_);
}

void SetCurStream(const StreamPyPtr &cur_stream) {
  MS_EXCEPTION_IF_NULL(cur_stream);
  MS_LOG(DEBUG) << "current_stream_id:" << cur_stream->stream_id();
  cur_stream->device_ctx()->device_res_manager_->SetCurrentStreamId(cur_stream->stream_id());
}

void Synchronize() {
  auto device_ctx = GetDeviceCtx();
  runtime::Pipeline::Get().WaitForward();
  device::MultiStreamController::GetInstance()->Refresh(device_ctx);
  (void)device::MultiStreamController::GetInstance()->SyncAllStreams(device_ctx);
}

StreamPyPtr CurrentStream() {
  auto device_ctx = GetDeviceCtx();
  auto current_stream_id = device_ctx->device_res_manager_->GetCurrentStreamId();
  MS_LOG(DEBUG) << "current_stream_id:" << current_stream_id;
  return std::make_shared<StreamPy>(device_ctx, current_stream_id);
}

StreamPyPtr DefaultStream() {
  auto device_ctx = GetDeviceCtx();
  const auto &default_stream_id = device_ctx->device_res_manager_->DefaultStream();
  return std::make_shared<StreamPy>(device_ctx, default_stream_id);
}

void RegStream(py::module *m) {
  (void)py::class_<StreamPy, std::shared_ptr<StreamPy>>(*m, "Stream")
    .def(py::init<int>())
    .def(py::init<int, int>())
    .def(py::init<const StreamPy &>())
    .def("query", &StreamPy::Query)
    .def("synchronize", &StreamPy::Synchronize)
    .def("__repr__", &StreamPy::ToStringRepr)
    .def("__eq__", &StreamPy::StreamEqual)
    .def_property_readonly("id", &StreamPy::stream_id)
    .def_property_readonly("device_name", &StreamPy::device_name)
    .def_property_readonly("device_id", &StreamPy::device_id);

  (void)m->def("set_cur_stream", &mindspore::hal::SetCurStream, "Set current stream");
  (void)m->def("synchronize", &mindspore::hal::Synchronize, "Synchronize all stream");
  (void)m->def("current_stream", &mindspore::hal::CurrentStream, "Get current stream");
  (void)m->def("default_stream", &mindspore::hal::DefaultStream, "Get default stream");
}
}  // namespace hal
}  // namespace mindspore
