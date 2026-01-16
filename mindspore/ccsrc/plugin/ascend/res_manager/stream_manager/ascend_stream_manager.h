/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_STREAM_MANAGER_ASCEND_STREAM_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_STREAM_MANAGER_ASCEND_STREAM_MANAGER_H_

#include <memory>
#include <vector>
#include <set>
#include <mutex>

#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/stream_manager/callback_thread.h"
#include "utils/hash_map.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
class ASCEND_RES_MANAGER_EXPORT AscendStreamMng {
 public:
  static AscendStreamMng &GetInstance();

  ~AscendStreamMng() = default;
  void Clear() {
#ifdef WITH_BACKEND
    for (auto iter = stream_call_backs_.begin(); iter != stream_call_backs_.end();) {
      aclrtStream stream = iter->first;
      iter++;
      UnRegCallback(stream, false);
    }
#endif
  }

  void ResetResource() {
    cur_stream_num_ = 0;
    cur_event_num_ = 0;
  }

  uint32_t ApplyNewStream() { return cur_stream_num_++; }

  uint32_t ApplyNewEvent() { return cur_event_num_++; }

  aclrtEvent ApplyRtEvent();
  aclrtEvent ApplyRtEventWithFlag(uint32_t flag);
  uint32_t GetRtEventId(const aclrtEvent &event) const;
  void DestroyAllRtEvents();

  void DeleteEvent();

  void DeleteStream();

  uint32_t GetCurAllocStreamId() const;

  uint32_t cur_stream_num() const { return cur_stream_num_; }

  uint32_t cur_event_num() const { return cur_event_num_; }

  void CreateStream(aclrtStream *stream, int32_t priority = 0);
  void CreateStream(size_t *stream_id, int32_t priority = 0);
  void RegCallback(aclrtStream stream);
  void UnRegCallback(aclrtStream stream, bool delete_item = true);
  void CreateStreamWithFlags(aclrtStream *stream, uint32_t flags, int32_t priority = 0);
  void CreateStreamWithFlags(size_t *stream_id, uint32_t flags, int32_t priority = 0);
  bool DestroyStream(size_t stream_id);
  bool DestroyAllStreams();
  bool ForceDestroyAllStreams();
  aclrtStream GetStream(size_t stream_id) const;
  bool SyncStream(size_t stream_id) const;
  bool SyncStream(aclrtStream stream) const;
  // 'sync_device' means whether calling 'aclrtSynchronizeDeviceWithTimeout' or 'aclrtSynchronizeStreamWithTimeout'.
  bool SyncAllStreams(bool sync_device = true) const;
  bool SyncNotDefaultStreams() const;
  // Sync all streams except the streams in except_streams.
  bool SyncExceptStreamsInList(const std::set<aclrtStream> &except_streams) const;
  size_t QueryStreamSize() const;
  bool QueryStream(size_t stream_id);
  size_t GetStreamId(void *stream_ptr);
  std::vector<uint32_t> GetStreamIds() const;
  void SetBusyStreamNum(uint32_t stream_num) { busy_stream_num_ = stream_num; }
  uint32_t GetBusyStreamNum() const { return busy_stream_num_; }
  void SetCopyToStream(aclrtStream stream) { copy_to_stream_ = stream; }
  void SetCopyInStream(aclrtStream stream) { copy_in_stream_ = stream; }
  void SetCopyOutStream(aclrtStream stream) { copy_out_stream_ = stream; }
  void SetForwardSendStream(aclrtStream stream) { forward_send_stream_ = stream; }
  void SetBackwardSendStream(aclrtStream stream) { backward_send_stream_ = stream; }
  void SetForwardRecvStream(aclrtStream stream) { forward_recv_stream_ = stream; }
  void SetBackwardRecvStream(aclrtStream stream) { backward_recv_stream_ = stream; }
  void SetRemoteCopyInStream(aclrtStream stream) { remote_copy_in_stream_ = stream; }
  void SetRemoteCopyOutStream(aclrtStream stream) { remote_copy_out_stream_ = stream; }
  aclrtStream GetCopyToStream() const { return copy_to_stream_; }
  aclrtStream GetCopyInStream() const { return copy_in_stream_; }
  aclrtStream GetCopyOutStream() const { return copy_out_stream_; }
  aclrtStream GetForwardSendStream() const { return forward_send_stream_; }
  aclrtStream GetBackwardSendStream() const { return backward_send_stream_; }
  aclrtStream GetForwardRecvStream() const { return forward_recv_stream_; }
  aclrtStream GetBackwardRecvStream() const { return backward_recv_stream_; }
  aclrtStream GetRemoteCopyInStream() const { return remote_copy_in_stream_; }
  aclrtStream GetRemoteCopyOutStream() const { return remote_copy_out_stream_; }

  void set_current_stream(size_t stream_id) { current_stream_id_ = stream_id; }
  size_t current_stream() const { return current_stream_id_; }

  void CreateDefaultStream();
  size_t default_stream_id() const;
  size_t communication_stream_id() const;
  aclrtStream default_stream() const;
  aclrtStream communication_stream() const;

  bool single_op_multi_stream_enable() const { return single_op_multi_stream_enable_; }
  void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
    single_op_multi_stream_enable_ = single_op_multi_stream_enable;
  }

  void enable_callback(bool is_enable_callback) { is_enable_callback_ = is_enable_callback; }
  bool is_enable_callback() { return is_enable_callback_; }

 private:
  // Count streams and events number in task sink scenario
  uint32_t cur_stream_num_{0};
  uint32_t cur_event_num_{0};

  // The max stream num on device ar a time
  uint32_t busy_stream_num_{0};

  // Ensure the thread safety for creating and destroying stream.
  std::mutex stream_mutex_;
  aclrtStream copy_to_stream_{nullptr};
  aclrtStream copy_in_stream_{nullptr};
  aclrtStream copy_out_stream_{nullptr};
  aclrtStream remote_copy_in_stream_{nullptr};
  aclrtStream remote_copy_out_stream_{nullptr};
  aclrtStream forward_send_stream_{nullptr};
  aclrtStream backward_send_stream_{nullptr};
  aclrtStream forward_recv_stream_{nullptr};
  aclrtStream backward_recv_stream_{nullptr};

  // all gpu CUDA streams including default_stream_.
  std::vector<void *> streams_;
  std::vector<aclrtEvent> events_{};

  // Currently using stream id.
  size_t current_stream_id_{0};

  // Default stream. We consider the first stream created as default stream.
  aclrtStream default_stream_{nullptr};
  size_t default_stream_id_{0};
  aclrtStream communication_stream_{nullptr};
  size_t communication_stream_id_{0};

  bool single_op_multi_stream_enable_{false};

  // Flag of registering callback or not, default value is false.
  // When multi streams are created, or gmem is enabled, this flag would change to ture.
  bool is_enable_callback_{false};
  // This vector used for simplify logic of tracing multi stream creates.
  std::vector<aclrtStream> callback_cached_streams_;
  mindspore::HashMap<aclrtStream, CallbackThreadPtr> stream_call_backs_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_STREAM_MANAGER_ASCEND_STREAM_MANAGER_H_
