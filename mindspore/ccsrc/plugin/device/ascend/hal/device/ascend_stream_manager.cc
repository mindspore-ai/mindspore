/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif
#include "external/acl/error_codes/rt_error_codes.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/hal/device/ascend_gmem_adapter.h"

namespace mindspore {
namespace device {
namespace ascend {
AscendStreamMng &AscendStreamMng::GetInstance() {
  static AscendStreamMng instance{};
  return instance;
}

void AscendStreamMng::DestroyAllRtEvents() {
  for (size_t i = 0; i < events_.size(); ++i) {
    if (events_[i] != nullptr) {
      auto rt_ret = aclrtDestroyEvent(events_[i]);
      if (rt_ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Call aclrtDestroyEvent failed, ret:" << rt_ret;
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

void AscendStreamMng::CreateStream(aclrtStream *stream, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  auto ret = aclrtCreateStreamWithConfig(stream, IntToUint(priority), (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = aclrtSetStreamFailureMode(*stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
  // If this is the first stream ever created, set it as default stream.
  if (streams_.size() == 1) {
    default_stream_ = *stream;
    default_stream_id_ = kIndex0;
  }
  RegCallback(*stream);
}

void AscendStreamMng::CreateStream(size_t *stream_id, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  aclrtStream stream;
  auto ret = aclrtCreateStreamWithConfig(&stream, IntToUint(priority), (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = aclrtSetStreamFailureMode(stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  *stream_id = streams_.size();
  (void)streams_.emplace_back(stream);
  RegCallback(stream);
}

void AscendStreamMng::RegCallback(aclrtStream stream) {
  MS_LOG(INFO) << "Register callback thread, stream : " << stream << ".";
  (void)callback_cached_streams_.emplace_back(stream);
  if (callback_cached_streams_.size() > 1 && !is_enable_callback_) {
    is_enable_callback_ = true;
  }
  if (!is_enable_callback_) {
    return;
  }
#ifdef WITH_BACKEND
  for (const auto &callback_cached_stream : callback_cached_streams_) {
    if (stream_call_backs_.count(callback_cached_stream) > 0) {
      MS_LOG(WARNING) << "Register callback thread failed, stream : " << callback_cached_stream
                      << " is already registered.";
      continue;
    }

    auto callback_thread = std::make_shared<CallbackThread>();
    callback_thread->create();
    auto ret = aclrtSubscribeReport(callback_thread->thread_, (aclrtStream)callback_cached_stream);
    if (!ret) {
      MS_LOG(INFO) << "Register callback thread success, stream : " << callback_cached_stream << ".";
      (void)stream_call_backs_.emplace(callback_cached_stream, callback_thread);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Register callback thread failed, stream : " << callback_cached_stream
                                 << ", ret : " << ret;
    }
  }
#endif
  callback_cached_streams_.clear();
}

void AscendStreamMng::UnRegCallback(aclrtStream stream) {
  MS_LOG(INFO) << "Unregister callback thread, stream : " << stream << ".";
  if (!is_enable_callback_) {
    return;
  }
#ifdef WITH_BACKEND
  if (stream_call_backs_.count(stream) == 0) {
    MS_LOG(WARNING) << "Unregister callback thread failed, stream : " << stream << " is not exist.";
    return;
  }
  auto callback_thread = stream_call_backs_.at(stream);
  // Cannot call aclrtUnSubscribeReport.
  callback_thread->cancel();
  stream_call_backs_.erase(stream);
#endif
}

void AscendStreamMng::CreateStreamWithFlags(aclrtStream *stream, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  auto ret = aclrtCreateStreamWithConfig(stream, IntToUint(priority), flags);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = aclrtSetStreamFailureMode(*stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
  RegCallback(*stream);
}

void AscendStreamMng::CreateStreamWithFlags(size_t *stream_id, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  aclrtStream stream;
  auto ret = aclrtCreateStreamWithConfig(&stream, IntToUint(priority), flags);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = aclrtSetStreamFailureMode(stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  *stream_id = streams_.size();
  (void)streams_.emplace_back(stream);
  RegCallback(stream);
}

aclrtEvent AscendStreamMng::ApplyRtEvent() {
  aclrtEvent rt_event = nullptr;
  auto ret = aclrtCreateEvent(&rt_event);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "aclrtCreateEvent failed, ret:" << ret;
  }
  (void)events_.emplace_back(rt_event);
  return rt_event;
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
  const auto ret = aclrtDestroyStream(streams_.at(stream_id));
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call aclrtDestroyStream, ret[" << ret << "]";
  }
  UnRegCallback(streams_.at(stream_id));
  streams_[stream_id] = nullptr;
  return true;
}

bool AscendStreamMng::DestroyAllStreams() {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  for (const auto &stream : streams_) {
    if (stream == nullptr) {
      continue;
    }
    const auto ret = aclrtDestroyStream(stream);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call aclrtDestroyStream, ret[" << ret << "]";
    }
    UnRegCallback(stream);
  }
  streams_.clear();
  return true;
}

aclrtStream AscendStreamMng::GetStream(size_t stream_id) const {
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

bool AscendStreamMng::SyncStream(aclrtStream stream) const {
  MS_EXCEPTION_IF_NULL(stream);
  auto RET = aclrtSynchronizeStreamWithTimeout(stream, -1);
  if (RET != ACL_ERROR_NONE && RET != ACL_ERROR_RT_AICORE_OVER_FLOW) {  // o for switch stream
    MS_LOG(ERROR) << "Call runtime aclrtSynchronizeStreamWithTimeout error.";
    return false;
  }
  if (RET == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    MS_LOG(WARNING) << "Call runtime aclrtSynchronizeStreamWithTimeout, the stream get overflow.";
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

bool AscendStreamMng::SyncNotDefaultStreams() const {
  bool res = true;
  for (size_t i = 0; i < streams_.size(); i++) {
    if (i != default_stream_id_ && !SyncStream(i)) {
      MS_LOG(ERROR) << "Failed to sync for ascend stream id: " << i;
      res = false;
    }
  }
  return res;
}

bool AscendStreamMng::SyncExceptStreamsInList(const std::set<aclrtStream> &except_streams) const {
  bool res = true;
  for (size_t i = 0; i < streams_.size(); i++) {
    if (except_streams.count(streams_[i]) > 0) {
      MS_LOG(DEBUG) << "Stream id:" << i << " is been synchronized.";
      continue;
    }
    if (!SyncStream(i)) {
      MS_LOG(ERROR) << "Failed to sync for ascend stream id: " << i;
      res = false;
    }
  }
  return res;
}

bool AscendStreamMng::QueryStream(size_t stream_id) {
  if (stream_id >= streams_.size()) {
    MS_LOG(EXCEPTION) << "Stream for stream id[" << stream_id << "] has not been created.";
  }
  const auto stream = streams_[stream_id];
  if (stream == nullptr) {
    MS_LOG(WARNING) << "Stream for stream id[" << stream_id << "] has been destroyed.";
    return false;
  }

  aclrtStreamStatus status;
  auto ret = aclrtStreamQuery(stream, &status);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to query completion status for stream id: " << stream_id;
  }
  return status == ACL_STREAM_STATUS_COMPLETE;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
