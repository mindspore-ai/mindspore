/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_MESSAGE_QUEUE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_MESSAGE_QUEUE_H_

#include <memory>
#include <string>

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/types.h>
#endif

#include "utils/status.h"

namespace mindspore::dataset {
#if !defined(_WIN32) && !defined(_WIN64)
const int kMsgQueuePermission = 0600;
const int kMsgQueueClosed = 2;
const int kMsgQueueInterrupted = 4;

// content is err status which is stored in err_msg_
const int32_t kWorkerErrorMsgSize = 4096;  // the max length of err msg which will be sent to main process

// indicate that master consumer(iterator/to_device) is finish
const int64_t kMasterReceiveBridgeOpFinishedMsg = 222;  // master -> worker, request mtype

// contest is Tensor(normal data / eoe / eof) which is stored in shared memory
const int64_t kWorkerSendDataMsg = 777;  // worker -> master, request mtype
const int64_t kMasterSendDataMsg = 999;  // master -> worker, response mtype

const int64_t kSubprocessReadyMsg = 555;   // when the subprocess is forked, the main process can continue to run
const int64_t kMainprocessReadyMsg = 666;  // the main process got message from subprocess, response to the subprocess

const int32_t kFourBytes = 4;

enum MessageState {
  kInit = 0,
  kRunning = 1,
  kReleased,
};

class MessageQueue {
 public:
  explicit MessageQueue(key_t key);

  MessageQueue(key_t key, int msg_queue_id);

  ~MessageQueue();

  void SetReleaseFlag(bool flag);

  void ReleaseQueue();

  Status GetOrCreateMessageQueueID();

  MessageState MessageQueueState();

  Status MsgSnd(int64_t mtype, int shm_id = -1, uint64_t shm_size = 0);

  Status MsgRcv(int64_t mtype);

  // wrapper the msgrcv
  int MsgRcv(int64_t mtype, int msgflg);

  // convert Status to err msg
  Status SerializeStatus(const Status &status);

  // convert Python Error to err msg
  Status SerializeStatus(const int32_t &status_code, const int32_t &line_of_code, const std::string &filename,
                         const std::string &err_desc);

  // convert err msg to Status
  Status DeserializeStatus();

  // get the err status flag
  bool GetErrorStatus();

  // clear the err status for ds.config.set_error_samples_mode(...) scenario
  void ClearErrMsg();

  // the below is the message content
  // kWorkerSendDataMsg, normal tensor from subprocess to main process
  // kMasterSendDataMsg, response from main process to subprocess
  int64_t mtype_;                      // the message type
  int32_t shm_id_;                     // normal Tensor, the shm id
  uint64_t shm_size_;                  // normal Tensor, the shm size
  char err_msg_[kWorkerErrorMsgSize];  // exception, the err msg from subprocess to main process

  key_t key_;             // message key
  int32_t msg_queue_id_;  // the msg queue id
  bool release_flag_;     // whether release the msg_queue_id_ when ~MessageQueue
  MessageState state_;    // whether the msg_queue_id_ had been released
};
#endif
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_MESSAGE_QUEUE_H_
