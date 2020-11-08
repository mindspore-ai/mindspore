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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_IPC_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_IPC_H_

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/msg.h>
#include <string>
#include <utility>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// A message queue structure between the parent and the child process
struct CacheMsgBuf {
  int64_t mtype;
  union {
    char mtext[1];
    struct {
      int32_t err_code;
      char err_msg[kSharedMessageSize];
    } status;
  } body;
};

class BaseIPC {
 public:
  BaseIPC() : remove_ipc_on_exit_(false) {}
  virtual ~BaseIPC() {}
  /// Indicate if we should remove the ipc resource on exit. Usually this is done by parent process.
  void RemoveResourcesOnExit() { remove_ipc_on_exit_ = true; }
  /// Copy constructors
  BaseIPC(const BaseIPC &rhs) : remove_ipc_on_exit_(false) {}
  BaseIPC &operator=(const BaseIPC &rhs) {
    if (&rhs != this) {
      remove_ipc_on_exit_ = false;
    }
    return *this;
  }
  /// Move constructors
  BaseIPC(BaseIPC &&rhs) noexcept : remove_ipc_on_exit_(rhs.remove_ipc_on_exit_) { rhs.remove_ipc_on_exit_ = false; }
  BaseIPC &operator=(BaseIPC &&rhs) noexcept {
    if (&rhs != this) {
      remove_ipc_on_exit_ = rhs.remove_ipc_on_exit_;
      rhs.remove_ipc_on_exit_ = false;
    }
    return *this;
  }

 protected:
  bool remove_ipc_on_exit_;
};

/// \brief This wraps a shared message for the communication between processes. It is used primarily
/// for starting and stopping a server.
class SharedMessage : public BaseIPC {
 public:
  using queue_id_t = int;
  SharedMessage() : msg_qid_(-1) {}
  explicit SharedMessage(queue_id_t qid) : msg_qid_(qid) {}
  ~SharedMessage() override;

  /// Copy constructors
  SharedMessage(const SharedMessage &rhs) : BaseIPC(rhs), msg_qid_(rhs.msg_qid_) {}
  SharedMessage &operator=(const SharedMessage &rhs) {
    if (&rhs != this) {
      msg_qid_ = rhs.msg_qid_;
      BaseIPC::operator=(rhs);
    }
    return *this;
  }
  /// Move constructors
  SharedMessage(SharedMessage &&rhs) noexcept : BaseIPC(std::move(rhs)) {
    msg_qid_ = rhs.msg_qid_;
    rhs.msg_qid_ = -1;
  }
  SharedMessage &operator=(SharedMessage &&rhs) noexcept {
    if (&rhs != this) {
      msg_qid_ = rhs.msg_qid_;
      rhs.msg_qid_ = -1;
      BaseIPC::operator=(std::move(rhs));
    }
    return *this;
  }

  /// Return the private id
  queue_id_t GetMsgQueueId() const { return msg_qid_; }

  /// \brief Create a private message queue
  Status Create();

  /// Send a Status object
  Status SendStatus(const Status &rc);

  /// Retrieve a Status object
  Status ReceiveStatus(Status *rc);

 private:
  queue_id_t msg_qid_;
};

/// \brief This wraps a shared memory for the communication between processes. It is used primarily
/// for transporting large tensor rows.
class SharedMemory : public BaseIPC {
 public:
  using shm_key_t = int;
  using shm_id_t = int;
  SharedMemory() : shm_id_(-1), shm_key_(-1), shmat_addr_(nullptr) {}
  explicit SharedMemory(shm_key_t public_key) : shm_id_(-1), shm_key_(public_key), shmat_addr_(nullptr) {}
  ~SharedMemory() override;
  /// Copy constructors
  SharedMemory(const SharedMemory &rhs)
      : BaseIPC(rhs), shm_id_(rhs.shm_id_), shm_key_(rhs.shm_key_), shmat_addr_(rhs.shmat_addr_) {}
  SharedMemory &operator=(const SharedMemory &rhs) {
    if (&rhs != this) {
      shm_id_ = rhs.shm_id_;
      shm_key_ = rhs.shm_key_;
      shmat_addr_ = rhs.shmat_addr_;
      BaseIPC::operator=(rhs);
    }
    return *this;
  }
  /// Move constructors
  SharedMemory(SharedMemory &&rhs) noexcept : BaseIPC(std::move(rhs)) {
    shm_id_ = rhs.shm_id_;
    shm_key_ = rhs.shm_key_;
    shmat_addr_ = rhs.shmat_addr_;
    rhs.shm_id_ = -1;
    rhs.shm_key_ = -1;
    rhs.shmat_addr_ = nullptr;
  }
  SharedMemory &operator=(SharedMemory &&rhs) noexcept {
    if (&rhs != this) {
      shm_id_ = rhs.shm_id_;
      shm_key_ = rhs.shm_key_;
      shmat_addr_ = rhs.shmat_addr_;
      rhs.shm_id_ = -1;
      rhs.shm_key_ = -1;
      rhs.shmat_addr_ = nullptr;
      BaseIPC::operator=(std::move(rhs));
    }
    return *this;
  }
  /// \brief Set the public key
  void SetPublicKey(key_t public_key) { shm_key_ = public_key; }

  /// \brief Retrieve the key
  shm_key_t GetKey() const { return shm_key_; }

  /// \brief This returns where we attach to the shared memory.
  /// \return Base address of the shared memory.
  const void *SharedMemoryBaseAddr() const { return shmat_addr_; }
  void *SharedMemoryBaseAddr() { return shmat_addr_; }

  /// \brief Attach to shared memory
  /// \return Status object
  Status Attach();

  /// Detach from shared memory
  /// \return Status object
  Status Detach();

  /// Create shared memory
  /// \return Status object
  Status Create(int64_t sz);

  /// Destroy shared memory
  /// \return Status object
  Status Destroy();

  /// \brief Return the shared memory id
  shm_id_t GetSharedMemoryId() const { return shm_id_; }

  /// \brief Get number of processes attached to the shared memory
  /// \return Status object
  Status GetNumAttached(int32_t *num);

 private:
  shm_id_t shm_id_;
  shm_key_t shm_key_;
  void *shmat_addr_;
};

/// \brief Generate a shared memory key using the tcp/ip port.
/// \note It must be called after the cache server generates the unix socket or ftok will fail.
/// \note Caller must check the return value. -1 means ftok failed.
/// \param[in] port
/// \param[out] err. If not null and ftok fails, this will contain the value of errno
/// \return key
Status PortToFtok(int port, SharedMemory::shm_key_t *);

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_IPC_H_
