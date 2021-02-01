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

#include "src/runtime/gpu/cuda/cuda_runtime.h"
#include <vector>
#include <mutex>
#include "include/errorcode.h"
#include "src/common/file_utils.h"

namespace mindspore::lite::cuda {

static std::mutex g_mtx;

bool CudaRuntime::initialized_ = false;
uint32_t CudaRuntime::instance_count_ = 0;
CudaRuntime *CudaRuntime::cuda_runtime_instance_ = nullptr;

CudaRuntime *CudaRuntime::GetInstance() {
  std::unique_lock<std::mutex> lck(g_mtx);
  static CudaRuntime vk_runtime;
  if (instance_count_ == 0) {
    cuda_runtime_instance_ = &vk_runtime;
    cuda_runtime_instance_->Init();
  }
  instance_count_++;
  return cuda_runtime_instance_;
}

void CudaRuntime::DeleteInstance() {
  std::unique_lock<std::mutex> lck(g_mtx);
  if (instance_count_ == 0) {
    MS_LOG(ERROR) << "No VulkanRuntime instance could delete!";
  }
  instance_count_--;
  if (instance_count_ == 0) {
    cuda_runtime_instance_->Uninit();
  }
}

CudaRuntime::CudaRuntime() {}

// Init will get platforms info, get devices info, create opencl context.
int CudaRuntime::Init() {
  if (initialized_) {
    return RET_OK;
  }

  initialized_ = true;
  MS_LOG(INFO) << "CudaRuntime init done!";

  return RET_OK;
}

int CudaRuntime::Uninit() {
  if (!initialized_) {
    return RET_OK;
  }
  initialized_ = false;
  return RET_OK;
}

CudaRuntime::~CudaRuntime() { Uninit(); }

const GpuInfo &CudaRuntime::GetGpuInfo() { return gpu_info_; }
bool CudaRuntime::GetFp16Enable() const { return true; }

}  // namespace mindspore::lite::cuda
