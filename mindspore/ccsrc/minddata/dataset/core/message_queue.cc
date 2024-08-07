/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/message_queue.h"

#include <string>

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
#if !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID)
MessageQueue::MessageQueue(key_t key, int msg_queue_id)
    : mtype_(0),
      shm_id_(-1),
      shm_size_(0),
      key_(key),
      msg_queue_id_(msg_queue_id),
      release_flag_(true),
      state_(State::kInit) {}

MessageQueue::~MessageQueue() {
  if (release_flag_ && msg_queue_id_ != -1 && msgget(key_, kMsgQueuePermission) == msg_queue_id_) {
    if (msgctl(msg_queue_id_, IPC_RMID, 0) == -1) {
      MS_LOG(ERROR) << "Delete msg queue id: " << msg_queue_id_ << " failed.";
    }
    state_ = State::kReleased;
    MS_LOG(INFO) << "Delete msg queue id: " << msg_queue_id_ << " success.";
    msg_queue_id_ = -1;
  }
}

void MessageQueue::SetReleaseFlag(bool flag) { release_flag_ = flag; }

Status MessageQueue::GetOrCreateMessageQueueID() {
  msg_queue_id_ = msgget(key_, kMsgQueuePermission);
  if (msg_queue_id_ < 0 && state_ != State::kReleased) {
    // create message queue id
    int msg_queue_id_ = msgget(key_, IPC_CREAT | kMsgQueuePermission);
    if (msg_queue_id_ < 0) {
      RETURN_STATUS_UNEXPECTED("Create send message by key: " + std::to_string(key_) +
                               " failed. Errno: " + std::to_string(errno));
    }
    MS_LOG(INFO) << "Create send message queue id: " << std::to_string(msg_queue_id_)
                 << " by key: " << std::to_string(key_) << " success.";
  }
  state_ = State::kRunning;
  return Status::OK();
}

Status MessageQueue::MsgSnd(int64_t mtype, int shm_id, uint64_t shm_size) {
  RETURN_IF_NOT_OK(GetOrCreateMessageQueueID());
  mtype_ = mtype;
  shm_id_ = shm_id;
  shm_size_ = shm_size;
  if (msg_queue_id_ >= 0 && msgsnd(msg_queue_id_, this, sizeof(MessageQueue), 0) != 0) {
    if (msgget(key_, kMsgQueuePermission) < 0) {
      MS_LOG(INFO) << "Main process is exit, msg_queue_id: " << std::to_string(msg_queue_id_) << " had been released.";
      return Status::OK();
    }
    RETURN_STATUS_UNEXPECTED("Exec msgsnd failed. Msg queue id: " + std::to_string(msg_queue_id_) +
                             ", mtype: " + std::to_string(mtype) + ", shm_id: " + std::to_string(shm_id) +
                             ", shm_size: " + std::to_string(shm_size));
  }
  MS_LOG(DEBUG) << "Exec msgsnd success, mtype: " << mtype << ", shm_id: " << shm_id << ", shm_size: " << shm_id;
  return Status::OK();
}

Status MessageQueue::MsgRcv(int64_t mtype) {
  if (msg_queue_id_ >= 0 && msgrcv(msg_queue_id_, this, sizeof(MessageQueue), mtype, 0) <= 0) {
    if (msgget(key_, kMsgQueuePermission) < 0) {
      MS_LOG(INFO) << "The msg_queue_id: " << std::to_string(msg_queue_id_) << " had been released.";
      if (errno == kMsgQueueClosed) {  // the message queue had been closed
        state_ = State::kReleased;
      }
    }
    RETURN_STATUS_UNEXPECTED("Exec msgrcv failed. Msg queue id: " + std::to_string(msg_queue_id_) +
                             ", mtype: " + std::to_string(mtype) + ", errno: " + std::to_string(errno));
  }
  MS_LOG(DEBUG) << "Exec msgrcv success, mtype: " << mtype << ", shm_id: " << shm_id_ << ", shm_size: " << shm_id_;
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
