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

#ifndef MINDSPORE_CCSRC_PYBIND_API_HAL_STREAM_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_HAL_STREAM_PY_H
#include <string>
#include <memory>
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace hal {
class StreamPy {
 public:
  StreamPy();
  explicit StreamPy(int priority);
  StreamPy(int priority, int stream_id);
  StreamPy(device::DeviceContext *device_ctx, const size_t &stream_id);
  explicit StreamPy(const StreamPy &stream);
  ~StreamPy();

  // Query if the event has completed
  bool Query();

  void Synchronize();

  device::DeviceContext *device_ctx() const {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    return device_ctx_;
  }

  std::string ToStringRepr() const;

  size_t stream_id() const { return stream_id_; }

  void *stream() const;

  bool StreamEqual(const std::shared_ptr<StreamPy> other_stream);

  std::string device_name() const {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    return device::GetDeviceNameByType(device_ctx_->device_context_key().device_type_);
  }

  uint32_t device_id() const {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    return device_ctx_->device_context_key().device_id_;
  }

 private:
  device::DeviceContext *device_ctx_;
  size_t stream_id_{0};
};
using StreamPyPtr = std::shared_ptr<StreamPy>;

void SetCurStream(const StreamPyPtr &cur_stream);

// Synchronize all stream
void Synchronize();

// Get current stream
StreamPyPtr CurrentStream();

StreamPyPtr DefaultStream();

StreamPyPtr CommunicationStream();
}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_HAL_STREAM_PY_H
