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

#include "minddata/dataset/engine/gnn/graph_shared_memory.h"

#include <string>
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
namespace gnn {

GraphSharedMemory::GraphSharedMemory(int64_t memory_size, key_t memory_key)
    : memory_size_(memory_size),
      memory_key_(memory_key),
      memory_ptr_(nullptr),
      memory_offset_(0),
      is_new_create_(false) {
  std::stringstream stream;
  stream << std::hex << memory_key_;
  memory_key_str_ = stream.str();
}

GraphSharedMemory::GraphSharedMemory(int64_t memory_size, const std::string &mr_file)
    : mr_file_(mr_file),
      memory_size_(memory_size),
      memory_key_(-1),
      memory_ptr_(nullptr),
      memory_offset_(0),
      is_new_create_(false) {}

GraphSharedMemory::~GraphSharedMemory() {
  if (is_new_create_) {
    (void)DeleteSharedMemory();
  }
}

Status GraphSharedMemory::CreateSharedMemory() {
  if (memory_key_ == -1) {
    // ftok to generate unique key
    auto realpath = FileUtils::GetRealPath(mr_file_.data());
    CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Get real path failed, path=" + mr_file_);
    memory_key_ = ftok(common::SafeCStr(realpath.value()), kGnnSharedMemoryId);
    CHECK_FAIL_RETURN_UNEXPECTED(memory_key_ != -1, "Failed to get key of shared memory. file_name:" + mr_file_);
    std::stringstream stream;
    stream << std::hex << memory_key_;
    memory_key_str_ = stream.str();
  }
  int shmflg = (0666 | IPC_CREAT | IPC_EXCL);
  Status s = SharedMemoryImpl(shmflg);
  if (s.IsOk()) {
    is_new_create_ = true;
    MS_LOG(INFO) << "Create shared memory success, key=0x" << memory_key_str_;
  } else {
    MS_LOG(WARNING) << "Shared memory with the same key may already exist, key=0x" << memory_key_str_;
    shmflg = (0666 | IPC_CREAT);
    s = SharedMemoryImpl(shmflg);
    if (!s.IsOk()) {
      RETURN_STATUS_UNEXPECTED("Create shared memory fao;ed, key=0x" + memory_key_str_);
    }
  }
  return Status::OK();
}

Status GraphSharedMemory::GetSharedMemory() {
  int shmflg = 0;
  RETURN_IF_NOT_OK(SharedMemoryImpl(shmflg));
  return Status::OK();
}

Status GraphSharedMemory::DeleteSharedMemory() {
  int shmid = shmget(memory_key_, 0, 0);
  CHECK_FAIL_RETURN_UNEXPECTED(shmid != -1, "Failed to get shared memory. key=0x" + memory_key_str_);
  int result = shmctl(shmid, IPC_RMID, 0);
  CHECK_FAIL_RETURN_UNEXPECTED(result != -1, "Failed to delete shared memory. key=0x" + memory_key_str_);
  return Status::OK();
}

Status GraphSharedMemory::SharedMemoryImpl(const int &shmflg) {
  // shmget returns an identifier in shmid
  CHECK_FAIL_RETURN_UNEXPECTED(memory_size_ >= 0, "Invalid memory size, should be greater than zero.");
  int shmid = shmget(memory_key_, memory_size_, shmflg);
  CHECK_FAIL_RETURN_UNEXPECTED(shmid != -1, "Failed to get shared memory. key=0x" + memory_key_str_);

  // shmat to attach to shared memory
  auto data = shmat(shmid, reinterpret_cast<void *>(0), 0);
  CHECK_FAIL_RETURN_UNEXPECTED(data != (char *)(-1), "Failed to address shared memory. key=0x" + memory_key_str_);
  memory_ptr_ = reinterpret_cast<uint8_t *>(data);

  return Status::OK();
}

Status GraphSharedMemory::InsertData(const uint8_t *data, int64_t len, int64_t *offset) {
  CHECK_FAIL_RETURN_UNEXPECTED(data, "Input data is nullptr.");
  CHECK_FAIL_RETURN_UNEXPECTED(len > 0, "Input len is invalid.");
  CHECK_FAIL_RETURN_UNEXPECTED(offset, "Input offset is nullptr.");

  std::lock_guard<std::mutex> lck(mutex_);
  CHECK_FAIL_RETURN_UNEXPECTED((memory_size_ - memory_offset_ >= len),
                               "Insufficient shared memory space to insert data.");
  if (EOK != memcpy_s(memory_ptr_ + memory_offset_, memory_size_ - memory_offset_, data, len)) {
    RETURN_STATUS_UNEXPECTED("Failed to insert data into shared memory.");
  }
  *offset = memory_offset_;
  memory_offset_ += len;
  return Status::OK();
}

Status GraphSharedMemory::GetData(uint8_t *data, int64_t data_len, int64_t offset, int64_t get_data_len) {
  CHECK_FAIL_RETURN_UNEXPECTED(data, "Input data is nullptr.");
  CHECK_FAIL_RETURN_UNEXPECTED(get_data_len > 0, "Input get_data_len is invalid.");
  CHECK_FAIL_RETURN_UNEXPECTED(data_len >= get_data_len, "Insufficient target address space.");

  CHECK_FAIL_RETURN_UNEXPECTED(memory_size_ >= get_data_len + offset,
                               "get_data_len is too large, beyond the space of shared memory.");
  if (EOK != memcpy_s(data, data_len, memory_ptr_ + offset, get_data_len)) {
    RETURN_STATUS_UNEXPECTED("Failed to insert data into shared memory.");
  }
  return Status::OK();
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
