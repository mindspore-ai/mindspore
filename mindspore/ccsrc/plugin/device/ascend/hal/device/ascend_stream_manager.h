/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_MANAGER_H_

#include <memory>
#include <vector>
#include <mutex>
#include "utils/hash_map.h"
#include "runtime/event.h"
#include "runtime/stream.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendStreamMng {
 public:
  static AscendStreamMng &GetInstance();

  void ResetResource() {
    cur_stream_num_ = 0;
    cur_event_num_ = 0;
  }

  uint32_t ApplyNewStream() { return cur_stream_num_++; }

  uint32_t ApplyNewEvent() { return cur_event_num_++; }

  rtEvent_t ApplyRtEvent();
  rtEvent_t ApplyRtEventWithFlag(uint32_t flag);
  uint32_t GetRtEventId(const rtEvent_t &event) const;
  void DestroyAllRtEvents();

  void DeleteEvent();

  void DeleteStream();

  uint32_t GetCurAllocStreamId() const;

  uint32_t cur_stream_num() const { return cur_stream_num_; }

  uint32_t cur_event_num() const { return cur_event_num_; }

  void CreateStream(rtStream_t *stream, int32_t priority = 0);
  void CreateStream(size_t *stream_id, int32_t priority = 0);
  void CreateStreamWithFlags(rtStream_t *stream, uint32_t flags, int32_t priority = 0);
  void CreateStreamWithFlags(size_t *stream_id, uint32_t flags, int32_t priority = 0);
  bool DestroyStream(size_t stream_id);
  bool DestroyAllStreams();
  rtStream_t GetStream(size_t stream_id) const;
  bool SyncStream(size_t stream_id) const;
  bool SyncStream(rtStream_t stream) const;
  bool SyncAllStreams() const;
  void SetBusyStreamNum(uint32_t stream_num) { busy_stream_num_ = stream_num; }
  uint32_t GetBusyStreamNum() const { return busy_stream_num_; }

 private:
  // Count streams and events number in task sink scenario
  uint32_t cur_stream_num_{0};
  uint32_t cur_event_num_{0};

  // The max stream num on device ar a time
  uint32_t busy_stream_num_{0};

  // Ensure the thread safety for creating and destroying stream.
  std::mutex stream_mutex_;

  // all gpu CUDA streams including default_stream_.
  std::vector<void *> streams_;
  std::vector<rtEvent_t> events_{};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_MANAGER_H_
