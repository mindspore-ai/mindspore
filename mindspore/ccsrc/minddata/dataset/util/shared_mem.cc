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
#include "minddata/dataset/util/shared_mem.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#include <atomic>
#include <cerrno>
#include <random>

namespace mindspore::dataset {
#if !defined(_WIN32) && !defined(_WIN64)
constexpr uint32_t O_CREX = O_CREAT | O_EXCL;
static constexpr int64_t kMapAllocAlignment = 64;

struct CountInfo {
  std::atomic<int32_t> refcount;
};

std::string make_filename() {
  static std::atomic<uint64_t> counter{0};
  static std::random_device rd;
  std::string handle = "/mindspore_";
  handle += std::to_string(getpid());
  handle += "_";
  handle += std::to_string(rd());
  handle += "_";
  handle += std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
  return handle;
}

SharedMem::SharedMem(const std::string &name, bool create, size_t size) {
  if (size < 0) {
    MS_EXCEPTION(RuntimeError) << "SharedMemory: size must be a positive integer.";
  }

  if (create) {
    flags_ = O_CREX | O_RDWR;
    if (size == 0) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: size must be a positive number different from zero.";
    }
  }

  if (name.empty() && !(flags_ & O_EXCL)) {
    MS_EXCEPTION(RuntimeError) << "SharedMemory: name can only be empty if create is true.";
  }

  if (name.empty()) {
    std::string file_name = make_filename();
    fd_ = shm_open(file_name.c_str(), flags_, mode_);
    if (fd_ == -1) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: shm_open failed with errno: " << errno;
    }
    name_ = file_name;
  } else {
    std::string file_name = "/" + name;
    fd_ = shm_open(file_name.c_str(), flags_, mode_);
    if (fd_ == -1) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: shm_open failed with errno: " << errno;
    }
    name_ = file_name;
  }

  if (create && size != 0) {
    size += kMapAllocAlignment;
    int ret = ftruncate(fd_, size);
    if (ret == -1) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: ftruncate failed with errno: " << errno;
    }
  }

  struct stat file_stat {};
  if (fstat(fd_, &file_stat) == -1) {
    ::close(fd_);
  }
  auto file_size = static_cast<size_t>(file_stat.st_size);
  buf_ = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (buf_ == MAP_FAILED) {
    MS_EXCEPTION(RuntimeError) << "SharedMemory: mmap failed with errno: " << errno;
  }
  auto *count_info = reinterpret_cast<CountInfo *>(buf_);
  if (create) {
    new (&count_info->refcount) std::atomic<int32_t>(1);
  } else {
    count_info->refcount++;
  }
  size_ = file_size;
}

SharedMem::~SharedMem() { Close(); }

void *SharedMem::Buf() { return static_cast<void *>(static_cast<char *>(buf_) + kMapAllocAlignment); }

std::string SharedMem::Name() const {
  std::string reported_name = name_;
  if (name_.at(0) == '/') {
    reported_name = name_.substr(1);
  }
  return reported_name;
}

size_t SharedMem::Size() const { return size_; }

void SharedMem::Incref() {
  auto *info = static_cast<CountInfo *>(buf_);
  ++info->refcount;
}

int SharedMem::Decref() {
  auto *info = static_cast<CountInfo *>(buf_);
  return --info->refcount == 0;
}

Status SharedMem::Close() {
  if (buf_ != nullptr) {
    auto *info = reinterpret_cast<CountInfo *>(buf_);
    if (--info->refcount == 0) {
      RETURN_IF_NOT_OK(Unlink());
    }

    int ret = munmap(buf_, size_);
    CHECK_FAIL_RETURN_UNEXPECTED(ret == 0, "SharedMemory: munmap failed with errno: " + std::to_string(errno));
    buf_ = nullptr;
  }

  if (fd_ >= 0) {
    int ret = ::close(fd_);
    CHECK_FAIL_RETURN_UNEXPECTED(ret == 0, "SharedMemory: close fd failed with errno: " + std::to_string(errno));
    fd_ = -1;
  }
  return Status::OK();
}

Status SharedMem::Unlink() {
  if (!name_.empty()) {
    int ret = shm_unlink(name_.c_str());
    CHECK_FAIL_RETURN_UNEXPECTED(ret == 0, "SharedMemory: shm_inlink failed with errno: " + std::to_string(errno));
  }
  return Status::OK();
}
#endif
}  // namespace mindspore::dataset
