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

#ifndef MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_STREAM_H_
#define MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace hal {
class PYNATIVE_UTILS_EXPORT Stream {
 public:
  Stream();
  explicit Stream(int priority);
  Stream(int priority, int stream_id);
  Stream(device::DeviceContext *device_ctx, const size_t &stream_id);
  explicit Stream(const Stream &stream);
  virtual ~Stream();

  // Query if the stream has completed all operations
  bool Query();

  void Synchronize();

  device::DeviceContext *device_ctx() const {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    return device_ctx_;
  }

  std::string ToStringRepr() const;

  size_t stream_id() const { return stream_id_; }

  void *stream() const;

  uintptr_t stream_ptr() const;

  bool StreamEqual(const std::shared_ptr<Stream> other_stream);

  std::string device_name() const {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    return device::GetDeviceNameByType(device_ctx_->device_context_key().device_type_);
  }

  uint32_t device_id() const {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    return device_ctx_->device_context_key().device_id_;
  }

 protected:
  device::DeviceContext *device_ctx_;
  size_t stream_id_{0};
};
using StreamPtr = std::shared_ptr<Stream>;

PYNATIVE_UTILS_EXPORT void SetCurStream(const StreamPtr &cur_stream);

// Synchronize all streams
PYNATIVE_UTILS_EXPORT void SynchronizeAllStreams();

// Get current stream
PYNATIVE_UTILS_EXPORT StreamPtr CurrentStream();

PYNATIVE_UTILS_EXPORT size_t CurrentStreamId();

PYNATIVE_UTILS_EXPORT StreamPtr DefaultStream();

PYNATIVE_UTILS_EXPORT StreamPtr CommunicationStream();

}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_STREAM_H_
