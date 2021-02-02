/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/cache/cache_ipc.h"
#include <sys/stat.h>

namespace mindspore {
namespace dataset {
Status PortToFtok(int port, SharedMemory::shm_key_t *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  key_t shmkey = -1;
  const std::string unix_path = PortToUnixSocketPath(port);
  shmkey = ftok(unix_path.data(), 'a');
  if (shmkey == (key_t)-1) {
    std::string errMsg = "Unable to create a ftok token. Errno = " + std::to_string(errno);
    return Status(errno == ENOENT ? StatusCode::kMDFileNotExist : StatusCode::kMDUnexpectedError, errMsg);
  }
  *out = shmkey;
  return Status::OK();
}

SharedMessage::~SharedMessage() {
  // Only remove the queue if we are asked to.
  if (remove_ipc_on_exit_ && msg_qid_ != -1) {
    // Remove the message que and never mind about the return code.
    (void)msgctl(msg_qid_, IPC_RMID, nullptr);
    msg_qid_ = -1;
  }
}

Status SharedMessage::Create() {
  CHECK_FAIL_RETURN_UNEXPECTED(msg_qid_ == -1, "Message queue already created");
  auto access_mode = S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP;
  msg_qid_ = msgget(IPC_PRIVATE, IPC_CREAT | IPC_EXCL | access_mode);
  if (msg_qid_ == -1) {
    std::string errMsg = "Unable to create a message queue. Errno = " + std::to_string(errno);
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  return Status::OK();
}

Status SharedMessage::SendStatus(const Status &rc) {
  CHECK_FAIL_RETURN_UNEXPECTED(msg_qid_ != -1, "Invalid message queue id");
  CacheMsgBuf msg{
    1,
  };
  msg.body.status.err_code = static_cast<int32_t>(rc.StatusCode());
  auto err = memcpy_s(msg.body.status.err_msg, kSharedMessageSize, rc.ToString().data(), rc.ToString().size());
  CHECK_FAIL_RETURN_UNEXPECTED(err == EOK, "memcpy_s failed. err = " + std::to_string(err));
  msg.body.status.err_msg[rc.ToString().size()] = '\0';
  err = msgsnd(msg_qid_, reinterpret_cast<void *>(&msg), sizeof(msg.body.status), IPC_NOWAIT);
  if (err == -1) {
    std::string errMsg = "Failed to call msgsnd. Errno = " + std::to_string(errno);
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  return Status::OK();
}

Status SharedMessage::ReceiveStatus(Status *rc) {
  RETURN_UNEXPECTED_IF_NULL(rc);
  CHECK_FAIL_RETURN_UNEXPECTED(msg_qid_ != -1, "Invalid message queue id");
  struct CacheMsgBuf msg {};
  auto err = msgrcv(msg_qid_, reinterpret_cast<void *>(&msg), sizeof(msg.body.status), 0, MSG_NOERROR);
  if (err == -1) {
    std::string errMsg = "Failed to call msgrcv. Errno = " + std::to_string(errno);
    RETURN_STATUS_UNEXPECTED(errMsg);
  }

  Status rc_recv(static_cast<StatusCode>(msg.body.status.err_code), msg.body.status.err_msg);
  *rc = std::move(rc_recv);
  return Status::OK();
}

SharedMemory::~SharedMemory() {
  if (shmat_addr_) {
    (void)Detach();
  }
  if (remove_ipc_on_exit_ && shm_id_ != -1) {
    // Remove the shared memory and never mind about the return code.
    Status rc = Destroy();
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc.ToString();
    }
  }
  shm_id_ = -1;
  shmat_addr_ = nullptr;
}

Status SharedMemory::Create(int64_t sz) {
  auto access_mode = S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP;
  shm_id_ = shmget(shm_key_, sz, IPC_CREAT | IPC_EXCL | access_mode);
  if (shm_id_ == -1) {
    RETURN_STATUS_UNEXPECTED("Shared memory creation failed. Errno " + std::to_string(errno));
  } else {
    shmat_addr_ = shmat(shm_id_, nullptr, 0);
    if (shmat_addr_ == reinterpret_cast<void *>(-1)) {
      RETURN_STATUS_UNEXPECTED("Shared memory attach failed. Errno " + std::to_string(errno));
    }
  }
  return Status::OK();
}

Status SharedMemory::Attach() {
  shm_id_ = shmget(shm_key_, 0, 0);
  if (shm_id_ == -1) {
    RETURN_STATUS_UNEXPECTED("Shmget failed. Errno " + std::to_string(errno));
  }
  shmat_addr_ = shmat(shm_id_, nullptr, 0);
  if (shmat_addr_ == reinterpret_cast<void *>(-1)) {
    RETURN_STATUS_UNEXPECTED("Shared memory attach failed. Errno " + std::to_string(errno));
  }
  return Status::OK();
}

Status SharedMemory::Detach() {
  if (shmat_addr_) {
    auto err = shmdt(shmat_addr_);
    if (err == -1) {
      RETURN_STATUS_UNEXPECTED("Shared memory detach failed. Errno " + std::to_string(errno));
    }
  }
  shmat_addr_ = nullptr;
  return Status::OK();
}

Status SharedMemory::Destroy() {
  // Remove the shared memory and never mind about the return code.
  auto err = shmctl(shm_id_, IPC_RMID, nullptr);
  if (err == -1) {
    std::string errMsg = "Unable to remove shared memory with id " + std::to_string(shm_id_);
    errMsg += ". Errno :" + std::to_string(errno);
    errMsg += "\nPlesae remove it manually using ipcrm -m command";
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  return Status::OK();
}

Status SharedMemory::GetNumAttached(int32_t *num) {
  RETURN_UNEXPECTED_IF_NULL(num);
  struct shmid_ds ds {};
  auto err = shmctl(shm_id_, IPC_STAT, &ds);
  if (err == -1) {
    std::string errMsg = "Unable to query shared memory with id " + std::to_string(shm_id_);
    errMsg += "\nPlease remove it manually using ipcrm -m command";
    RETURN_STATUS_UNEXPECTED(errMsg);
  }
  *num = ds.shm_nattch;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
