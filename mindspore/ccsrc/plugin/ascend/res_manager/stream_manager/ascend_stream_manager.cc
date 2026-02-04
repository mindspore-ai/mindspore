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

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

#include <string>
#include "utils/log_adapter.h"
#include "acl/error_codes/rt_error_codes.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr size_t kIndex0 = 0;
}
AscendStreamMng &AscendStreamMng::GetInstance() {
  static AscendStreamMng instance{};
  return instance;
}

void AscendStreamMng::DestroyAllRtEvents() {
  for (size_t i = 0; i < events_.size(); ++i) {
    if (events_[i] != nullptr) {
      auto rt_ret = CALL_ASCEND_API(aclrtDestroyEvent, events_[i]);
      if (rt_ret != ACL_SUCCESS) {
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
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, stream, IntToUint(priority),
                             (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, *stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
  RegCallback(*stream);
}

void AscendStreamMng::CreateStream(size_t *stream_id, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  aclrtStream stream;
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, &stream, IntToUint(priority),
                             (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
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
    auto ret = CALL_ASCEND_API(aclrtSubscribeReport, callback_thread->thread_, (aclrtStream)callback_cached_stream);
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

void AscendStreamMng::UnRegCallback(aclrtStream stream, bool delete_item) {
  MS_LOG(INFO) << "Unregister callback thread, stream : " << stream << ".";
  if (!is_enable_callback_) {
    return;
  }
#ifdef WITH_BACKEND
  if (stream_call_backs_.count(stream) == 0) {
    MS_LOG(INFO) << "Unregister callback thread failed, stream : " << stream << " is not exist.";
    return;
  }
  auto callback_thread = stream_call_backs_.at(stream);
  // Cannot call aclrtUnSubscribeReport.
  callback_thread->cancel();
  if (delete_item) {
    stream_call_backs_.erase(stream);
  }
#endif
}

void AscendStreamMng::CreateStreamWithFlags(aclrtStream *stream, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, stream, IntToUint(priority), flags);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, *stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
  RegCallback(*stream);
}

void AscendStreamMng::CreateStreamWithFlags(size_t *stream_id, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  aclrtStream stream;
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, &stream, IntToUint(priority), flags);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  *stream_id = streams_.size();
  (void)streams_.emplace_back(stream);
  RegCallback(stream);
}

aclrtEvent AscendStreamMng::ApplyRtEvent() {
  aclrtEvent rt_event = nullptr;
  // Use ex api of event, so that no limits on event total size.
  uint32_t flag = ACL_EVENT_SYNC;
  auto ret = CALL_ASCEND_API(aclrtCreateEventExWithFlag, &rt_event, flag);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtCreateEventExWithFlag failed, ret : " << ret << ".";
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
  const auto ret = CALL_ASCEND_API(aclrtDestroyStream, streams_.at(stream_id));
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtDestroyStream, ret[" << ret << "]";
  }
  UnRegCallback(streams_.at(stream_id));
  streams_[stream_id] = nullptr;
  if (communication_stream_id_ == stream_id) {
    communication_stream_ = nullptr;
  }
  if (default_stream_id_ == stream_id) {
    default_stream_ = nullptr;
  }

  return true;
}

bool AscendStreamMng::ForceDestroyAllStreams() {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  for (const auto &stream : streams_) {
    if (stream == nullptr) {
      continue;
    }
    const auto ret = CALL_ASCEND_API(aclrtDestroyStreamForce, stream);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Call aclrtDestroyStream, ret[" << ret << "]";
    }
    UnRegCallback(stream);
  }
  streams_.clear();
  default_stream_ = nullptr;
  communication_stream_ = nullptr;
  return true;
}

bool AscendStreamMng::DestroyAllStreams() {
  std::lock_guard<std::mutex> lock_streams(stream_mutex_);
  for (const auto &stream : streams_) {
    if (stream == nullptr) {
      continue;
    }
    const auto ret = CALL_ASCEND_API(aclrtDestroyStream, stream);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Call aclrtDestroyStream, ret[" << ret << "]";
    }
    UnRegCallback(stream);
  }
  streams_.clear();
  default_stream_ = nullptr;
  communication_stream_ = nullptr;
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
  MS_LOG(DEBUG) << "Sync stream: " << stream;
  auto RET = ACL_SUCCESS;
  try {
    GilReleaseWithCheck gil_release;
    MS_VLOG(VL_RUNTIME_FRAMEWORK_STREAM) << "Begin sync stream:" << stream;
    RET = CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream, -1);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_STREAM) << "End sync stream:" << stream;
    if (RET != ACL_SUCCESS && RET != ACL_ERROR_RT_AICORE_OVER_FLOW) {  // o for switch stream
      if (!mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
        MS_LOG(ERROR) << "Call runtime aclrtSynchronizeStreamWithTimeout error."
                      << "Please do the following three things to confirm whether it is caused by the "
                      << "execution failure of a certain operator.\n"
                      << "    1.Set mindspore.runtime.launch_blocking() at the beginning of your python script.\n"
                      << "    2.Run again your python script.\n"
                      << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
                      << "Now you will get the certain failed op detailed infos.";

      } else {
        MS_LOG(ERROR) << "Has set launch blocking, but synchronous stream still failed. Please save the "
                         "complete log information to further identify the specific error cause.";
      }
      return false;
    }
  } catch (const std::exception &e) {
    if (!mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
      MS_LOG(ERROR) << "Sync stream failed. " << e.what()
                    << "Please do the following three things to confirm whether it is caused by the "
                    << "execution failure of a certain operator.\n"
                    << "    1.Set mindspore.runtime.launch_blocking() at the beginning of your python script.\n"
                    << "    2.Run again your python script.\n"
                    << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
                    << "Now you will get the certain failed op detailed infos.";
    } else {
      MS_LOG(ERROR) << "Has set launch blocking, but synchronous stream still failed. Please save the "
                       "complete log information to further identify the specific error cause.";
    }
    return false;
  }
  if (RET == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    MS_LOG(WARNING) << "Call runtime aclrtSynchronizeStreamWithTimeout, the stream get overflow.";
  }
  return true;
}

bool AscendStreamMng::SyncAllStreams(bool sync_device) const {
  auto RET = ACL_ERROR_NONE;
  try {
    GilReleaseWithCheck gil_release;
    if (sync_device) {
      // According to CANN, we need to set timeout to 2 hours for aclrtSynchronizeDeviceWithTimeout.
      int timeout = 7200000;
      RET = CALL_ASCEND_API(aclrtSynchronizeDeviceWithTimeout, timeout);
      if (RET != ACL_ERROR_NONE && RET != ACL_ERROR_RT_AICORE_OVER_FLOW) {
        if (!mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
          MS_LOG(ERROR) << "Call runtime aclrtSynchronizeDeviceWithTimeout error."
                        << "Please do the following three things to confirm whether it is caused by the "
                        << "execution failure of a certain operator.\n"
                        << "    1.Set mindspore.runtime.launch_blocking() at the beginning of your python script.\n"
                        << "    2.Run again your python script.\n"
                        << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
                        << "Now you will get the certain failed op detailed infos.";
        } else {
          MS_LOG(ERROR) << "Has set launch blocking, but synchronous stream still failed. Please save the "
                           "complete log information to further identify the specific error cause.";
        }
        return false;
      }
    } else {
      for (size_t i = 0; i < streams_.size(); i++) {
        const auto stream = streams_[i];
        if (stream != nullptr && !SyncStream(stream)) {
          MS_LOG(ERROR) << "SyncStream for stream id " << i << " failed.";
          return false;
        }
      }
    }
  } catch (const std::exception &e) {
    std::string sync_method = sync_device ? "aclrtSynchronizeDeviceWithTimeout" : "aclrtSynchronizeStreamWithTimeout";
    if (!mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
      MS_LOG(ERROR) << sync_method << " failed. " << e.what()
                    << "Please do the following three things to confirm whether it is caused by the "
                    << "execution failure of a certain operator.\n"
                    << "    1.Set mindspore.runtime.launch_blocking() at the beginning of your python script.\n"
                    << "    2.Run again your python script.\n"
                    << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
                    << "Now you will get the certain failed op detailed infos.";
    } else {
      MS_LOG(ERROR) << "Has set launch blocking, but synchronous stream still failed. Please save the "
                       "complete log information to further identify the specific error cause.";
    }
    return false;
  }
  if (RET == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    std::string sync_method = sync_device ? "aclrtSynchronizeDeviceWithTimeout" : "aclrtSynchronizeStreamWithTimeout";
    if (!mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
      MS_LOG(WARNING) << "Call runtime " << sync_method << ", the stream get overflow."
                      << "Please do the following three things to confirm whether it is caused by the "
                      << "execution failure of a certain operator.\n"
                      << "    1.Set mindspore.runtime.launch_blocking() at the beginning of your python script.\n"
                      << "    2.Run again your python script.\n"
                      << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
                      << "Now you will get the certain failed op detailed infos.";
    } else {
      MS_LOG(WARNING) << "Has set launch blocking, but synchronous stream still failed. Please save the "
                         "complete log information to further identify the specific error cause.";
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

size_t AscendStreamMng::QueryStreamSize() const { return streams_.size(); }

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
  auto ret = CALL_ASCEND_API(aclrtStreamQuery, stream, &status);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to query completion status for stream id: " << stream_id;
  }
  return status == ACL_STREAM_STATUS_COMPLETE;
}

size_t AscendStreamMng::GetStreamId(void *stream_ptr) {
  auto iter = std::find(streams_.begin(), streams_.end(), stream_ptr);
  if (iter == streams_.end()) {
    MS_LOG(EXCEPTION) << "Failed to find stream_ptr in streams_, stream_ptr:" << stream_ptr;
  }

  return LongToSize(std::distance(streams_.begin(), iter));
}

std::vector<uint32_t> AscendStreamMng::GetStreamIds() const {
  std::vector<uint32_t> stream_ids;
  for (size_t i = 0; i < streams_.size(); i++) {
    if (streams_[i] != nullptr) {
      (void)stream_ids.emplace_back(static_cast<uint32_t>(i));
    }
  }
  return stream_ids;
}

void AscendStreamMng::CreateDefaultStream() {
  if (default_stream_ == nullptr) {
    CreateStream(&default_stream_id_);
    MS_LOG(INFO) << "Create ascend default stream, stream id: " << default_stream_id_;
    default_stream_ = GetStream(default_stream_id_);
    MS_EXCEPTION_IF_NULL(default_stream_);
  } else {
    MS_LOG(INFO) << "The default compute stream is already created, skip.";
  }

  if (communication_stream_ == nullptr) {
    CreateStream(&communication_stream_id_);
    MS_LOG(INFO) << "Create ascend communication stream, stream id: " << communication_stream_id_;
    communication_stream_ = GetStream(communication_stream_id_);
    MS_EXCEPTION_IF_NULL(communication_stream_);
  } else {
    MS_LOG(INFO) << "The default communication stream is already created, skip.";
  }
}

size_t AscendStreamMng::default_stream_id() const {
  if (default_stream_ == nullptr) {
    MS_LOG(EXCEPTION) << "The default stream is not created";
  }
  return default_stream_id_;
}
size_t AscendStreamMng::communication_stream_id() const {
  if (communication_stream_ == nullptr) {
    MS_LOG(EXCEPTION) << "The communication stream is not created";
  }
  return communication_stream_id_;
}
aclrtStream AscendStreamMng::default_stream() const { return default_stream_; }
aclrtStream AscendStreamMng::communication_stream() const { return communication_stream_; }

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
