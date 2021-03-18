/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_GPU_RUNTIME_H_
#define MINDSPORE_LITE_SRC_GPU_RUNTIME_H_
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/runtime/allocator.h"
#include "schema/gpu_cache_generated.h"

namespace mindspore::lite::gpu {

enum GpuType { OTHER = 0, ADRENO = 1, MALI = 2, MALI_T = 3, MALI_G = 4 };
struct GpuInfo {
  GpuType type = OTHER;
  int model_num = 0;
  float version = 0;
  uint64_t global_memery_cachesize{0};
  uint64_t global_memery_size{0};
  uint64_t max_alloc_size{0};
  uint32_t max_work_group_size{1};
  uint32_t compute_units{0};
  uint32_t max_freq{0};
  uint32_t image_pitch_align{0};
  std::vector<size_t> max_work_item_sizes;
  bool support_fp16{false};
  bool support_svm{false};
};
enum class GpuBackendType { OPENCL = 0, CUDA = 1, VULKAN = 2 };
class DevKey {
 public:
  std::string name{""};
};
class GpuContext {
 public:
  GpuBackendType type;
};
class GpuDevice {
 public:
  GpuDevice();
  ~GpuDevice();
};
class DevKernel {
 public:
  void *data{nullptr};
};
class GpuAllocator : public mindspore::Allocator {};
class GpuRuntime {
 public:
  GpuRuntime() {}
  virtual ~GpuRuntime() {}
  GpuRuntime(const GpuRuntime &) = delete;
  GpuRuntime &operator=(const GpuRuntime &) = delete;

  virtual int Init() { return RET_ERROR; }
  virtual int Uninit() { return RET_ERROR; }
  virtual const GpuInfo &GetGpuInfo() = 0;
  virtual bool GetFp16Enable() const = 0;

  uint64_t GetGlobalMemSize() const { return gpu_info_.global_memery_size; }
  uint64_t GetMaxAllocSize() const { return gpu_info_.max_alloc_size; }
  const std::vector<size_t> &GetWorkItemSize() const { return gpu_info_.max_work_item_sizes; }

 protected:
  // gpu hal native defines
  std::unordered_map<std::string, DevKernel *> dev_kernels_;
  GpuContext *context_{nullptr};
  GpuDevice *device_{nullptr};
  GpuInfo gpu_info_;

 private:
};
template <class T>
class GpuRuntimeWrapper {
 public:
  GpuRuntimeWrapper() { gpu_runtime_ = T::GetInstance(); }
  ~GpuRuntimeWrapper() { T::DeleteInstance(); }
  GpuRuntimeWrapper(const GpuRuntimeWrapper &) = delete;
  GpuRuntimeWrapper &operator=(const GpuRuntimeWrapper &) = delete;
  T *GetInstance() { return gpu_runtime_; }

 private:
  T *gpu_runtime_{nullptr};
};

}  // namespace mindspore::lite::gpu
#endif  // MINDSPORE_LITE_SRC_GPU_RUNTIME_H_
