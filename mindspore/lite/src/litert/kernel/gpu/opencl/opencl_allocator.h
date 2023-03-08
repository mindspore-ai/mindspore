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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_GPU_OPENCL_OPENCL_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_GPU_OPENCL_OPENCL_ALLOCATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "src/litert/inner_allocator.h"
#include "CL/cl2.hpp"

namespace mindspore::lite::opencl {
// OpenCL memory type, SHARED only valid on Mali devices.
enum class MemType : char { BUF, IMG, SHARED, GLTexture };
#define UNLOCK_AND_RETURN_NULL(condition, ptr) \
  do {                                         \
    if (condition) {                           \
      UnLock();                                \
      return (ptr);                            \
    }                                          \
  } while (0)

class OpenCLRuntime;

struct ImageSize {
  size_t width = 0;
  size_t height = 0;
  size_t dtype = CL_FLOAT;
  bool operator==(const struct ImageSize &other) const {
    return width == other.width && height == other.height && dtype == other.dtype;
  }
};

class OpenCLAllocator : public mindspore::Allocator {
 public:
  explicit OpenCLAllocator(OpenCLRuntime *ocl_runtime);
  ~OpenCLAllocator() override;
  void SetContext(const AllocatorContext &ctx);
  void *Malloc(size_t size, MemType type) { return _Malloc(type, nullptr, size); }

  // malloc shared
  void *Malloc(size_t size) override { return _Malloc(MemType::SHARED, nullptr, size); }
  void *Malloc(size_t weight, size_t height, DataType type) override;
  // malloc buffer
  void *Malloc(size_t size, void *data) { return _Malloc(MemType::BUF, data, size); }
  // malloc image
  void *Malloc(const ImageSize &img_size, void *data = nullptr) { return _Malloc(MemType::IMG, data, 0, img_size); }
  void Free(void *ptr) override;
  int RefCount(void *ptr) override;
  int SetRefCount(void *ptr, int ref_count) override;
  int DecRefCount(void *ptr, int ref_count) override;
  int IncRefCount(void *ptr, int ref_count) override;
  size_t total_size();

  void Clear();
  cl::Image2D *GetImage(void *host_ptr);
  void *GetOpenclMemPtr(void *buffer, MemType *type, bool force_buffer = false);
  void *MapBuffer(void *host_ptr, int flags, void *command_queue = nullptr, bool sync = true);
  int UnmapBuffer(void *host_ptr, void *command_queue = nullptr);
  MemType GetMemType(void *host_ptr);
  int GetImageSize(void *host_ptr, ImageSize *img_size);
  void *Prepare(void *ptr) override {
    if (ptr != nullptr) {
      ptr = MapBuffer(ptr, CL_MAP_READ | CL_MAP_WRITE, nullptr, true);
    }
    return ptr;
  }
  bool IsOverSize() { return is_oversize_; }

 private:
  void Lock();
  void UnLock();
  void *MinimumFit(MemType mem_type, size_t size, const ImageSize &img_size);
  void *_Malloc(MemType mem_type, void *data, size_t size = 0, const ImageSize &img_size = ImageSize());
  void *CreateBuffer(size_t size, void *data, size_t flags, cl::Buffer **buffer);
  int CreateImage2D(size_t size, const ImageSize &img_size, void *data, size_t flags, bool is_map, cl::Buffer **buffer,
                    cl::Image2D **image, void **host_ptr);
  int GetImgDtypeSize(const ImageSize &img_size);
  template <typename T>
  void ClearMemList(T *list);

 private:
  OpenCLRuntime *ocl_runtime_{nullptr};
  std::mutex lock;
  struct MemBuf {
    std::atomic_int ref_count_ = 0;
    size_t size_{0};
    void *device_ptr_{nullptr};
    void *host_ptr_{nullptr};
    void *image_ptr_{nullptr};
    MemType mem_type_{MemType::BUF};
    ImageSize img_size_;
    bool map_flags_{false};
  };

  // <membuf->buf, membuf>
  std::unordered_map<void *, MemBuf *> allocated_list_;
  std::multimap<size_t, MemBuf *> free_list_;
  uint64_t total_size_{0};
  // 6 is empirical value
  int shift_factor_ = 6;
  bool lock_flag_ = true;
  bool is_oversize_{false};
};
}  // namespace mindspore::lite::opencl

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_GPU_OPENCL_OPENCL_ALLOCATOR_H_
