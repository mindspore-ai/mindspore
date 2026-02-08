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
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <atomic>
#include <cerrno>
#include <random>

#if !defined(_WIN32) && !defined(_WIN64)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {

MS_CORE_API std::string NewShareMemoryHandle();

// MapAllocator contains information of cpu shared memory. MapAllocator is part of DeviceAddress,
// so put this class under dir device_address.
class MS_CORE_API MapAllocator {
 public:
  MapAllocator(const std::string &name, bool create, int fd, size_t size);

  void *Alloc(size_t size);

  bool Free(void *base_ptr_);

  const char *filename() const { return filename_.c_str(); }

  int fd() const { return fd_; }

  size_t size() const { return size_; }

  int flags() const { return flags_; }

  ~MapAllocator() = default;

 protected:
  std::string filename_;
  bool create_;
  int fd_ = -1;
  size_t size_;
  bool closed_ = false;
  int flags_ = 0;
};
using MapAllocatorPtr = std::unique_ptr<MapAllocator>;

}  // namespace mindspore
