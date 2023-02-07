/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
AscendStreamMng &AscendStreamMng::GetInstance() {
  static AscendStreamMng instance{};
  return instance;
}

rtEvent_t AscendStreamMng::ApplyRtEvent() {
  auto rt_resource = std::make_shared<rtEvent_t>();
  auto ret = rtEventCreate(rt_resource.get());
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtEventCreate failed, ret:" << ret;
  }
  (void)events_.emplace_back(*rt_resource);
  return *rt_resource;
}

rtEvent_t AscendStreamMng::ApplyRtEventWithFlag(uint32_t flag) {
  rtEvent_t rt_event = nullptr;
  auto ret = rtEventCreateWithFlag(&rt_event, flag);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtEventCreateWithFlag failed, ret:" << ret;
  }
  (void)events_.emplace_back(rt_event);
  return rt_event;
}

uint32_t AscendStreamMng::GetRtEventId(const rtEvent_t &event) const {
  uint32_t rt_event_id = 0;
  auto rt_ret = rtGetEventID(event, &rt_event_id);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtGetEventID failed, ret:" << rt_ret;
  }
  return rt_event_id;
}

void AscendStreamMng::DestroyAllRtEvents() {
  for (size_t i = 0; i < events_.size(); ++i) {
    if (events_[i] != nullptr) {
      auto rt_ret = rtEventDestroy(events_[i]);
      if (rt_ret != RT_ERROR_NONE) {
        MS_LOG(ERROR) << "Call rtEventDestroy failed, ret:" << rt_ret;
      }
    }
  }
  events_.clear();
}

void AscendStreamMng::DeleteEvent() {
  if (cur_event_num_ == 0) {
    MS_LOG(WARNING) << "total event num is 0, no event to delete";
  } else {
    --cur_event_num_;
  }
}

void AscendStreamMng::DeleteStream() {
  if (cur_stream_num_ == 0) {
    MS_LOG(WARNING) << " total stream num is 0, no stream to delete";
  } else {
    --cur_stream_num_;
  }
}

uint32_t AscendStreamMng::GetCurAllocStreamId() const {
  if (cur_stream_num_ == 0) {
    MS_LOG(EXCEPTION) << "stream nums is 0, no stream id should be get";
  }
  return cur_stream_num_ - 1;
}

void AscendStreamMng::CreateStream(rtStream_t *stream, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  const auto ret = rtStreamCreate(stream, priority);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
}

void AscendStreamMng::CreateStream(size_t *stream_id, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  rtStream_t stream;
  const auto ret = rtStreamCreate(&stream, priority);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  *stream_id = streams_.size();
  (void)streams_.emplace_back(stream);
}

void AscendStreamMng::CreateStreamWithFlags(rtStream_t *stream, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  const auto ret = rtStreamCreateWithFlags(stream, priority, flags);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
}

void AscendStreamMng::CreateStreamWithFlags(size_t *stream_id, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  rtStream_t stream;
  const auto ret = rtStreamCreateWithFlags(&stream, priority, flags);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  *stream_id = streams_.size();
  (void)streams_.emplace_back(stream);
}

bool AscendStreamMng::DestroyStream(size_t stream_id) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  if (stream_id >= streams_.size()) {
    MS_LOG(ERROR) << "Ascend stream not found for stream id " << stream_id;
    return false;
  }
  if (streams_.at(stream_id) == nullptr) {
    MS_LOG(WARNING) << "Ascend stream hsa been destroyed for stream id " << stream_id;
    return true;
  }
  const auto ret = rtStreamDestroy(streams_.at(stream_id));
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtStreamDestroy, ret[" << ret << "]";
  }
  streams_[stream_id] = nullptr;
  return true;
}

bool AscendStreamMng::DestroyAllStreams() {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  for (const auto &stream : streams_) {
    const auto ret = rtStreamDestroy(stream);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rtStreamDestroy, ret[" << ret << "]";
    }
  }
  streams_.clear();
  return true;
}

rtStream_t AscendStreamMng::GetStream(size_t stream_id) const {
  if (stream_id >= streams_.size()) {
    MS_LOG(DEBUG) << "Stream for stream id[" << stream_id << "] not found, return nullptr.";
    return nullptr;
  }
  return streams_[stream_id];
}

bool AscendStreamMng::SyncStream(size_t stream_id) const {
  if (stream_id >= streams_.size()) {
    MS_LOG(EXCEPTION) << "Stream for stream id[" << stream_id << "] has not been created.";
  }
  const auto stream = streams_[stream_id];
  if (stream == nullptr) {
    MS_LOG(WARNING) << "Stream for stream id[" << stream_id << "] has been destroyed.";
    return false;
  }
  return SyncStream(stream);
}

bool AscendStreamMng::SyncStream(rtStream_t stream) const {
  MS_EXCEPTION_IF_NULL(stream);
  if (rtStreamSynchronize(stream) != RT_ERROR_NONE) {  // o for switch stream
    MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    return false;
  }
  return true;
}

bool AscendStreamMng::SyncAllStreams() const {
  for (size_t i = 0; i < streams_.size(); ++i) {
    const auto stream = streams_[i];
    if (stream != nullptr && !SyncStream(stream)) {
      MS_LOG(ERROR) << "SyncStream for stream id " << i << " failed.";
      return false;
    }
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
