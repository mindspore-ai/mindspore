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
std::string GenerateShmName() {
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

SharedMem::SharedMem(const std::string &name, bool create, int fd, size_t size)
    : name_(name), create_(create), fd_(fd), size_(size) {
  if (name_.empty()) {
    MS_EXCEPTION(RuntimeError) << "SharedMemory: name can not be empty.";
  }

  if (!create_ && fd_ < 0) {
    MS_EXCEPTION(RuntimeError) << "SharedMemory: fd must be non-negative when create is false, but got: " << fd_;
  }

  if (size_ <= 0) {
    MS_EXCEPTION(RuntimeError) << "SharedMemory: size must be a positive integer, but got: " << size_;
  }

  if (create_) {
    mode_t mode = 0600;
    if ((fd_ = shm_open(name_.c_str(), O_RDWR | O_CREAT | O_EXCL, mode)) == -1) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: shm_open failed with errno: " << errno;
    }
  }

  struct stat file_stat {};
  if (fstat(fd_, &file_stat) == -1) {
    errno_t fstat_errno = errno;
    if (create) {
      ::close(fd_);
    }
    MS_EXCEPTION(RuntimeError) << "SharedMemory: fstat failed with errno: " << fstat_errno;
  }

  if (size_ > static_cast<size_t>(file_stat.st_size)) {
    if (ftruncate(fd_, static_cast<off_t>(size_)) == -1) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: ftruncate failed with errno: " << errno;
    }
  }

  if ((buf_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0)) == MAP_FAILED) {
    buf_ = nullptr;
    MS_EXCEPTION(RuntimeError) << "SharedMemory: mmap failed with errno: " << errno;
  }

  if (create_) {
    if (shm_unlink(name_.c_str()) == -1) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: shm_unlink failed with errno: " + std::to_string(errno);
    }
    MS_LOG(INFO) << "Shared memory " << name_ << " has been created, fd: " << fd_ << ", size: " << size_;
  } else {
    MS_LOG(INFO) << "Attach to shared memory " << name_ << ", fd: " << fd_ << ", size: " << size_;
  }
}

SharedMem::~SharedMem() { Close(); }

void *SharedMem::Buf() { return buf_; }

std::string SharedMem::Name() const { return name_; }

int32_t SharedMem::Fd() const { return fd_; }

size_t SharedMem::Size() const { return size_; }

void SharedMem::Close() {
  if (buf_ != nullptr) {
    if (munmap(buf_, size_) != 0) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: munmap failed with errno: " << errno;
    }
    buf_ = nullptr;
  }

  if (fd_ >= 0) {
    if (::close(fd_) != 0) {
      MS_EXCEPTION(RuntimeError) << "SharedMemory: close fd failed with errno: " << errno;
    }
    fd_ = -1;
  }
  MS_LOG(INFO) << "Shared memory " << name_ << " has been closed.";
}
#endif
}  // namespace mindspore::dataset
