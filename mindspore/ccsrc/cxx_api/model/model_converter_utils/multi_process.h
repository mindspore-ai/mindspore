/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_CXXAPI_MULTI_PROCESS_H
#define MINDSPORE_CCSRC_CXXAPI_MULTI_PROCESS_H
#include <iostream>
#include <functional>
#include "include/api/status.h"

namespace mindspore {
struct MessageFlag {
  uint64_t heartbeat = 0;
  uint64_t stop = false;
  uint64_t msg_len = 0;
  uint64_t msg_total_len = 0;
  uint64_t read_ready_flag = false;
  uint64_t read_finish_flag = false;
};

class MultiProcess;
using ProcessFuncCall = std::function<Status(MultiProcess *multi_process)>;
using CreateBufferCall = std::function<uint8_t *(size_t msg_len)>;

class MultiProcess {
 public:
  MultiProcess();
  ~MultiProcess();

  Status MainProcess(ProcessFuncCall parent_process, ProcessFuncCall child_process);
  Status SendMsg(const void *buffer, uint64_t msg_len);
  Status ReceiveMsg(CreateBufferCall create_buffer_call);

 private:
  uint8_t *shmat_addr_ = nullptr;
  uint8_t *shmat_data_addr_ = nullptr;
  uint64_t shmat_data_max_size_ = 0;
  uint64_t memory_size_ = 0;

  bool peer_stopped_ = false;
  bool stopped_ = false;
  MessageFlag *send_msg_ = nullptr;
  MessageFlag *receive_msg_ = nullptr;

  static void HeartbeatThreadFunc(MultiProcess *multi_process);
  void HeartbeatThreadFuncInner();
  Status ParentProcess(ProcessFuncCall parent_process);
  void ChildProcess(ProcessFuncCall child_process);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXXAPI_MULTI_PROCESS_H
