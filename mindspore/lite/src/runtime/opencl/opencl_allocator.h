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

#ifndef MINDSPORE_LITE_SRC_OPENCL_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_OPENCL_ALLOCATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "src/runtime/allocator.h"

namespace mindspore::lite::opencl {

#define MS_HOST_BUFFER 0
#define MS_CL_BUFFER (1 << 1)
#define MS_CL_IMAGE2D (1 << 2)
typedef int32_t OpenCLMemoryType;

struct OpenclMemory {
  void *host_ptr{nullptr};
  void *device_ptr{nullptr};
  OpenCLMemoryType mem_type{MS_HOST_BUFFER | MS_CL_BUFFER};
};

class OpenCLAllocator : public Allocator {
 public:
  OpenCLAllocator();
  ~OpenCLAllocator() override;
  void SetContext(const AllocatorContext &ctx) override;
  void *Malloc(size_t size) override;
  void Free(void *ptr) override;
  size_t GetTotalSize() override;
  void Clear() override;
  void *GetDeviceBuffer(void *buffer);
  void *MapBuffer(void *host_ptr, int flags, void *command_queue = nullptr, bool sync = true);
  int UnmapBuffer(void *host_ptr, void *command_queue = nullptr);

 private:
  void Lock();
  void UnLock();
  struct MemBuf {
    size_t size_;
    void *device_ptr_;
    void *host_ptr_;
  };

  std::mutex lock;
  // <membuf->buf, membuf>
  std::unordered_map<void *, MemBuf *> allocated_list_;
  std::multimap<size_t, MemBuf *> free_list_;
  // 6 is empirical value
  int shift_factor_ = 6;
  bool lock_flag_ = false;
};

}  // namespace mindspore::lite::opencl

#endif  // MINDSPORE_LITE_SRC_OPENCL_ALLOCATOR_H_

