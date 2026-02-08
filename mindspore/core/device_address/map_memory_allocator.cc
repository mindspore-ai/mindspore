/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <atomic>
#include <cerrno>
#include <random>

#if !defined(_WIN32) && !defined(_WIN64)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include "device_address/map_memory_allocator.h"

namespace mindspore {

std::string NewShareMemoryHandle() {
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

MapAllocator::MapAllocator(const std::string &name, bool create, int fd, size_t size)
    : filename_(name), create_(create), fd_(fd), size_(size) {}

void *MapAllocator::Alloc(size_t size) {
  void *base_ptr_ = nullptr;
#if defined(_WIN32) || defined(_WIN64)
  MS_EXCEPTION(RuntimeError) << "MapAllocator don't support Win32 or Win64 platform.";
#else
  if (create_ && filename_.empty()) {
    MS_EXCEPTION(RuntimeError) << "MapAllocator: when create is true, filename_ can not be empty.";
  }

  if (!create_ && fd_ < 0) {
    MS_EXCEPTION(RuntimeError) << "MapAllocator: fd must be non-negative when create is false, but got: " << fd_;
  }

  if (size == 0) {
    MS_EXCEPTION(RuntimeError) << "MapAllocator: size must be a positive integer, but got: " << size;
  }

  if (create_) {
    mode_t mode = 0600;
    if ((fd_ = shm_open(filename_.c_str(), O_RDWR | O_CREAT | O_EXCL, mode)) == -1) {
      MS_EXCEPTION(RuntimeError) << "MapAllocator: shm_open failed with errno: " << errno;
    }
  }

  struct stat file_stat {};
  if (fstat(fd_, &file_stat) == -1) {
    errno_t fstat_errno = errno;
    if (create_) {
      ::close(fd_);
    }
    MS_EXCEPTION(RuntimeError) << "MapAllocator: fstat failed with errno: " << fstat_errno;
  }

  if (size > static_cast<size_t>(file_stat.st_size)) {
    MS_LOG(INFO) << "MapAllocator: size:" << size << ", file_stat.st_size:" << static_cast<size_t>(file_stat.st_size);
    if (ftruncate(fd_, static_cast<off_t>(size)) == -1) {
      MS_EXCEPTION(RuntimeError) << "MapAllocator: ftruncate failed with errno: " << errno;
    }
  }

  if ((base_ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0)) == MAP_FAILED) {
    base_ptr_ = nullptr;
    MS_EXCEPTION(RuntimeError) << "MapAllocator: mmap failed with errno: " << errno;
  }

  if (create_) {
    if (shm_unlink(filename_.c_str()) == -1) {
      MS_EXCEPTION(RuntimeError) << "MapAllocator: shm_unlink failed with errno: " + std::to_string(errno);
    }
    MS_LOG(INFO) << "MapAllocator filename:" << filename_ << " has been created, fd: " << fd_ << ", size: " << size;
  } else {
    MS_LOG(INFO) << "MapAllocator attach to shared memory " << filename_ << ", fd: " << fd_ << ", size: " << size;
  }
#endif
  return base_ptr_;
}

bool MapAllocator::Free(void *base_ptr_) {
  MS_LOG(INFO) << "MapAllocator free enter.base_ptr_ is not null:" << (base_ptr_ != nullptr) << ", fd:" << fd_;
  if (closed_) {
    MS_LOG(INFO) << "MapAllocator has been closed.filename:" << filename_;
    return true;
  }
  closed_ = true;
#if defined(_WIN32) || defined(_WIN64)
  MS_EXCEPTION(RuntimeError) << "MapAllocator.Free() don't support Win32 or Win64 platform.";
#else
  if (base_ptr_ != nullptr) {
    if (munmap(base_ptr_, size_) != 0) {
      MS_EXCEPTION(RuntimeError) << "MapAllocator munmap failed.";
    }
    base_ptr_ = nullptr;
  }

  if (fd_ >= 0) {
    if (::close(fd_) != 0) {
      MS_EXCEPTION(RuntimeError) << "MapAllocator close fd failed.";
    }
    fd_ = -1;
  }
#endif
  MS_LOG(INFO) << "MapAllocator filename:" << filename_ << " closed successfully.";
  return true;
}

}  // namespace mindspore
