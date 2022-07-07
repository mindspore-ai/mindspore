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

#include "cxx_api/model/model_converter_utils/multi_process.h"
#include <unistd.h>
#include <sys/wait.h>
#include <algorithm>
#include <vector>
#include <thread>
#include "mindspore/core/utils/log_adapter.h"
#include "cxx_api/model/model_converter_utils/shared_memory.h"

namespace mindspore {
namespace {
constexpr uint64_t kSharedMemorySize = 100ull << 20;  // 100 MB
constexpr timespec kOneMillisecond = {
  0,                  // 0 seconds
  1 * 1000L * 1000L,  // And 1 ms
};

constexpr timespec kOneHundredMilliseconds = {
  0,                    // 0 seconds
  100 * 1000L * 1000L,  // And 100 ms
};
}  // namespace

MultiProcess::MultiProcess() = default;

MultiProcess::~MultiProcess() = default;

Status MultiProcess::MainProcess(const ProcessFuncCall &parent_process, const ProcessFuncCall &child_process) {
  MS_EXCEPTION_IF_NULL(parent_process);
  MS_EXCEPTION_IF_NULL(child_process);
  Status ret;
  memory_size_ = kSharedMemorySize;  // 100 MB
  SharedMemory shared_memory;
  ret = shared_memory.Create(memory_size_);
  if (ret != kSuccess) {
    MS_LOG_ERROR << "Create shared memory failed";
    return ret;
  }
  pid_t pid = fork();
  if (pid < 0) {
    shared_memory.Destroy();
    MS_LOG_ERROR << "Fork process to convert model failed";
    return kMEFailed;
  }
  ret = shared_memory.Attach();
  if (ret != kSuccess) {
    MS_LOG_ERROR << "Process attach shared memory failed, pid " << pid;
    return ret;
  }
  shmat_addr_ = shared_memory.GetSharedMemoryAddr();
  if (shmat_addr_ == nullptr) {
    MS_LOG_ERROR << "Get shared memory failed";
    return ret;
  }
  constexpr size_t kMsgStructNum = 2;
  shmat_data_addr_ = shmat_addr_ + sizeof(MessageFlag) * kMsgStructNum;
  shmat_data_max_size_ =
    memory_size_ - (reinterpret_cast<uintptr_t>(shmat_data_addr_) - reinterpret_cast<uintptr_t>(shmat_addr_));
  MS_LOG_INFO << "Shm addr " << reinterpret_cast<uintptr_t>(shmat_addr_);
  if (pid == 0) {
    ChildProcess(child_process);
    shared_memory.Detach();
    MS_LOG_INFO << "Model converter: child process sleep waiting for exit signal.";
    while (1) {
      // waiting for signal
    }
  } else {  // parent process
    ret = ParentProcess(parent_process);
    shared_memory.Detach();

    MS_LOG_INFO << "Model converter: parent process kills child of fork.";
    (void)kill(pid, SIGKILL);
    constexpr uint32_t kMaxLoopCount = 5;
    bool child_exited = false;
    for (uint32_t i = 0; i < kMaxLoopCount; ++i) {
      int status;
      if (waitpid(pid, &status, WNOHANG) == pid) {
        MS_LOG(INFO) << "Child process " << pid << " exits success.";
        child_exited = true;
        break;
      }
      (void)sleep(1);
    }
    if (!child_exited) {
      MS_LOG(WARNING) << "Child process " << pid << " has been killed but waitpid failed.";
    }
    shared_memory.Destroy();
  }
  return ret;
}

Status MultiProcess::ParentProcess(const ProcessFuncCall &parent_process) {
  auto parent_msg = reinterpret_cast<MessageFlag *>(shmat_addr_);
  auto child_msg = reinterpret_cast<MessageFlag *>(shmat_addr_ + sizeof(MessageFlag));
  send_msg_ = parent_msg;
  receive_msg_ = child_msg;
  std::thread heartbeat_thread(MultiProcess::HeartbeatThreadFunc, this);
  Status ret;
  try {
    ret = parent_process(this);
    if (ret != kSuccess) {
      MS_LOG_ERROR << "Parent process process failed";
    }
  } catch (const std::runtime_error &ex) {
    MS_LOG_ERROR << "Catch parent process runtime error: " << ex.what();
    ret = kMEFailed;
  }
  stopped_ = true;
  send_msg_->stop = 1;
  heartbeat_thread.join();
  return ret;
}

void MultiProcess::ChildProcess(const ProcessFuncCall &child_process) {
  auto parent_msg = reinterpret_cast<MessageFlag *>(shmat_addr_);
  auto child_msg = reinterpret_cast<MessageFlag *>(shmat_addr_ + sizeof(MessageFlag));
  send_msg_ = child_msg;
  receive_msg_ = parent_msg;
  std::thread heartbeat_thread(MultiProcess::HeartbeatThreadFunc, this);
  try {
    MS_EXCEPTION_IF_NULL(child_process);
    auto ret = child_process(this);
    if (ret != kSuccess) {
      MS_LOG_ERROR << "Child process process failed";
    }
  } catch (const std::runtime_error &ex) {
    MS_LOG_ERROR << "Catch child process runtime error: " << ex.what();
  }
  stopped_ = true;
  send_msg_->stop = 1;
  heartbeat_thread.join();
}

Status MultiProcess::SendMsg(const void *buffer, uint64_t msg_len) {
  MS_EXCEPTION_IF_NULL(buffer);
  MS_LOG_INFO << "Start to send message to peer process, msg len " << msg_len;
  send_msg_->msg_total_len = msg_len;
  uint64_t cur_offset = 0;
  while (msg_len > cur_offset) {
    uint64_t sub_msg_len = std::min(msg_len - cur_offset, shmat_data_max_size_);
    if (sub_msg_len == 0) {
      MS_LOG(ERROR) << "Invalid message len " << sub_msg_len;
      return kMEFailed;
    }
    auto ret =
      memcpy_s(shmat_data_addr_, shmat_data_max_size_, static_cast<const uint8_t *>(buffer) + cur_offset, sub_msg_len);
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed, ret = " << ret;
      return kMEFailed;
    }
    cur_offset += sub_msg_len;

    send_msg_->msg_len = sub_msg_len;
    send_msg_->read_finish_flag = 0;
    send_msg_->read_ready_flag = 1;
    MS_LOG_INFO << "Send start " << cur_offset << ", msg len " << sub_msg_len << ", total len " << msg_len;
    while (!send_msg_->read_finish_flag && !peer_stopped_) {
      (void)nanosleep(&kOneMillisecond, nullptr);  // 1ms
    }
    if (peer_stopped_) {
      if (!send_msg_->read_finish_flag) {
        return kMEFailed;
      }
      break;
    }
    MS_LOG_INFO << "Send end " << cur_offset << ", msg len " << sub_msg_len << ", total len " << msg_len;
  }
  MS_LOG_INFO << "End to send message to peer process, msg len " << msg_len;
  return kSuccess;
}

Status MultiProcess::ReceiveMsg(const CreateBufferCall &create_buffer_call) const {
  uint64_t cur_offset = 0;
  uint8_t *msg_buffer = nullptr;
  uint64_t msg_len = 0;
  do {
    MS_LOG_INFO << "Receive start from " << cur_offset;
    while (!receive_msg_->read_ready_flag && !peer_stopped_) {
      (void)nanosleep(&kOneMillisecond, nullptr);  // 1ms
    }
    if (peer_stopped_) {
      return kMEFailed;
    }
    if (msg_buffer == nullptr) {
      msg_len = receive_msg_->msg_total_len;
      msg_buffer = create_buffer_call(msg_len);
    }
    MS_EXCEPTION_IF_NULL(msg_buffer);
    auto ret = memcpy_s(msg_buffer + cur_offset, msg_len - cur_offset, shmat_data_addr_, receive_msg_->msg_len);
    if (ret != EOK) {
      MS_LOG(INFO) << "memcpy_s failed, ret = " << ret;
      return kMEFailed;
    }
    cur_offset += receive_msg_->msg_len;
    receive_msg_->read_ready_flag = 0;
    receive_msg_->read_finish_flag = 1;
    MS_LOG_INFO << "Receive end, current length " << cur_offset << ", total length " << msg_len << std::endl;
  } while (msg_len > cur_offset);
  return kSuccess;
}

void MultiProcess::HeartbeatThreadFunc(MultiProcess *multi_process) { multi_process->HeartbeatThreadFuncInner(); }

void MultiProcess::HeartbeatThreadFuncInner() {
  constexpr uint64_t kOvertime = 1024;
  uint64_t last_beat_cnt = 0;
  uint64_t repeat_cnt = 0;
  while (!stopped_) {
    if (receive_msg_->stop) {
      peer_stopped_ = true;
      MS_LOG_WARNING << "Peer stopped";
      break;
    }
    uint64_t heartbeat_gap = receive_msg_->heartbeat - last_beat_cnt;
    if (heartbeat_gap > 0 && heartbeat_gap < kOvertime) {
      last_beat_cnt = receive_msg_->heartbeat;
      repeat_cnt = 0;
    } else {
      repeat_cnt++;
      if (repeat_cnt > 30) {  // 30*100ms = 3s no reply
        peer_stopped_ = true;
        MS_LOG_WARNING << "Peer stopped";
        break;
      }
    }
    send_msg_->heartbeat += 1;
    (void)nanosleep(&kOneHundredMilliseconds, nullptr);  // sleep 100 ms
  }
}
}  // namespace mindspore
