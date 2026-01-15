/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/stream/stream.h"
#include <utility>
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "utils/ms_context.h"
#include "utils/current_state.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace hal {
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

Stream::Stream() {
  device_ctx_ = GetDeviceCtx();
  device_ctx_->device_res_manager_->set_single_op_multi_stream_enable(true);
}

Stream::Stream(int priority) {
  device_ctx_ = GetDeviceCtx();
  device_ctx_->device_res_manager_->CreateStreamWithPriority(&stream_id_, priority);
  MS_LOG(DEBUG) << "stream_id:" << stream_id_ << ", priority:" << priority;
  device_ctx_->device_res_manager_->set_single_op_multi_stream_enable(true);
  auto &controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_ctx_->device_context_key().device_type_);
  controller->Refresh();
}

Stream::Stream(int priority, int stream_id) {
  device_ctx_ = GetDeviceCtx();
  stream_id_ = IntToSize(stream_id);
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  if (device_ctx_->device_res_manager_->GetStream(stream_id) == nullptr) {
    MS_EXCEPTION(ValueError) << "stream_id:" << stream_id << " is not exist";
  }
  device_ctx_->device_res_manager_->set_single_op_multi_stream_enable(true);
}

Stream::Stream(device::DeviceContext *device_ctx, const size_t &stream_id)
    : device_ctx_(device_ctx), stream_id_(stream_id) {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  device_ctx_->device_res_manager_->set_single_op_multi_stream_enable(true);
}

Stream::Stream(const Stream &stream) : device_ctx_(stream.device_ctx()), stream_id_(stream.stream_id()) {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  device_ctx_->device_res_manager_->set_single_op_multi_stream_enable(true);
}

Stream::~Stream() { device_ctx_ = nullptr; }

bool Stream::Query() {
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  runtime::Pipeline::Get().WaitForward();
  return device_ctx_->device_res_manager_->QueryStream(stream_id_);
}

void Stream::Synchronize() {
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  runtime::Pipeline::Get().WaitForward();
  auto &controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_ctx_->device_context_key().device_type_);
  controller->Refresh();
  (void)controller->SyncStream(stream_id_);
}

std::string Stream::ToStringRepr() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(device_ctx_);
  buf << "Stream(device_name=" << device::GetDeviceNameByType(device_ctx_->device_context_key().device_type_)
      << ", device_id:" << std::to_string(device_ctx_->device_context_key().device_id_)
      << ", stream id:" << std::to_string(stream_id_) << ")";
  return buf.str();
}

void *Stream::stream() const {
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  return device_ctx_->device_res_manager_->GetStream(stream_id_);
}

uintptr_t Stream::stream_ptr() const {
  MS_LOG(DEBUG) << "stream_id:" << stream_id_;
  MS_EXCEPTION_IF_NULL(device_ctx_);
  MS_EXCEPTION_IF_NULL(device_ctx_->device_res_manager_);
  void *ptr = device_ctx_->device_res_manager_->GetStream(stream_id_);
  return reinterpret_cast<uintptr_t>(ptr);
}

bool Stream::StreamEqual(const std::shared_ptr<Stream> other_stream) {
  MS_EXCEPTION_IF_NULL(other_stream);

  MS_LOG(DEBUG) << "stream info:" << ToStringRepr() << " other_stream info:" << other_stream->ToStringRepr();
  return (stream_id_ == other_stream->stream_id()) &&
         (device_ctx()->device_context_key().device_type_ ==
          other_stream->device_ctx()->device_context_key().device_type_) &&
         (device_ctx()->device_context_key().device_id_ == other_stream->device_ctx()->device_context_key().device_id_);
}

void SetCurStream(const StreamPtr &cur_stream) {
  MS_EXCEPTION_IF_NULL(cur_stream);
  MS_LOG(DEBUG) << "current_stream_id:" << cur_stream->stream_id();
  MS_EXCEPTION_IF_NULL(cur_stream->device_ctx());
  MS_EXCEPTION_IF_NULL(cur_stream->device_ctx()->device_res_manager_);
  cur_stream->device_ctx()->device_res_manager_->SetCurrentStreamId(cur_stream->stream_id());
  CurrentStream::set_id(cur_stream->stream_id());
}

void SynchronizeAllStreams() {
  auto device_ctx = GetDeviceCtx();
  runtime::Pipeline::Get().WaitForward();
  MsException::Instance().CheckException();
  auto &controller =
    device::DeviceContextManager::GetInstance().GetMultiStreamController(device_ctx->device_context_key().device_type_);
  controller->Refresh();
  (void)controller->SyncAllStreams();
}

StreamPtr CurrentStream() {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  auto current_stream_id = mindspore::CurrentStream::id();
  MS_LOG(DEBUG) << "current_stream_id:" << current_stream_id;
  return std::make_shared<Stream>(device_ctx, current_stream_id);
}

size_t CurrentStreamId() {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  auto current_stream_id = mindspore::CurrentStream::id();
  MS_LOG(DEBUG) << "current_stream_id:" << current_stream_id;
  return current_stream_id;
}

StreamPtr DefaultStream() {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  const auto &default_stream_id = device_ctx->device_res_manager_->DefaultStream();
  return std::make_shared<Stream>(device_ctx, default_stream_id);
}

StreamPtr CommunicationStream() {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  const auto &comm_stream_id = device_ctx->device_res_manager_->GetCommunicationStreamID();
  return std::make_shared<Stream>(device_ctx, comm_stream_id);
}

}  // namespace hal
}  // namespace mindspore
