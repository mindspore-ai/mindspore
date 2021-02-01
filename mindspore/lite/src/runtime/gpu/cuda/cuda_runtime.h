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

#ifndef MINDSPORE_LITE_SRC_CUDA_RUNTIME_H_
#define MINDSPORE_LITE_SRC_CUDA_RUNTIME_H_
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include "src/common/log_adapter.h"
#include "src/runtime/gpu/gpu_runtime.h"
#include "schema/gpu_cache_generated.h"

using mindspore::lite::gpu::GpuInfo;
using mindspore::lite::gpu::GpuRuntime;
using mindspore::lite::gpu::GpuRuntimeWrapper;

namespace mindspore::lite::cuda {

class CudaRuntime : public GpuRuntime {
 public:
  friend GpuRuntimeWrapper<CudaRuntime>;
  ~CudaRuntime() override;
  CudaRuntime(const CudaRuntime &) = delete;
  CudaRuntime &operator=(const CudaRuntime &) = delete;

  int Init() override;
  int Uninit() override;
  const GpuInfo &GetGpuInfo() override;
  bool GetFp16Enable() const override;

  static CudaRuntime *GetInstance();
  static void DeleteInstance();

 private:
  CudaRuntime();

 private:
  static bool initialized_;
  static uint32_t instance_count_;
  static CudaRuntime *cuda_runtime_instance_;
};
}  // namespace mindspore::lite::cuda
#endif  // MINDSPORE_LITE_SRC_CUDA_RUNTIME_H_
