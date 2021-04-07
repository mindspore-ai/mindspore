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

#include "src/inner_context.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#ifdef SUPPORT_NPU
#include "src/runtime/agent/npu/npu_manager.h"
#endif

namespace mindspore::lite {
InnerContext::InnerContext(const Context *context) {
  this->allocator = context->allocator;
  this->thread_num_ = context->thread_num_;
  this->device_list_.clear();
  for (auto &device_ctx : context->device_list_) {
    this->device_list_.push_back(device_ctx);
  }
}

#if SUPPORT_NPU
InnerContext::InnerContext(const Context *context, NPUManager *npu_manager) {
  this->allocator = context->allocator;
  this->thread_num_ = context->thread_num_;
  this->device_list_.clear();
  for (auto &device_ctx : context->device_list_) {
    if (device_ctx.device_type_ == DT_CPU) {
      // npu server would use one core so we don't bind core to avoid competition.
      auto cpu_ctx = device_ctx;
      cpu_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = NO_BIND;
      this->device_list_.push_back(cpu_ctx);
    } else {
      this->device_list_.push_back(device_ctx);
    }
  }
  this->npu_manager_ = npu_manager;
}
#endif

int InnerContext::Init() {
  if (RET_OK != this->IsValid()) {
    MS_LOG(ERROR) << "Context is not valid";
    return RET_NOT_SUPPORT;
  }
  if (this->thread_pool_ == nullptr && this->IsCpuEnabled()) {
    this->thread_pool_ =
      CreateLiteThreadPool(this->thread_num_, this->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_);
    if (this->thread_pool_ == nullptr) {
      MS_LOG(ERROR) << "Create ThreadPool failed";
      return RET_NULL_PTR;
    }
  }
  if (this->allocator == nullptr) {
    this->allocator = mindspore::Allocator::Create();
    if (this->allocator == nullptr) {
      MS_LOG(ERROR) << "Create Allocator failed";
      return RET_NULL_PTR;
    }
  }
  if (IsNpuEnabled()) {
    MS_LOG(DEBUG) << "NPU enabled.";
  }
  if (IsGpuEnabled()) {
    MS_LOG(DEBUG) << "GPU enabled.";
  }
  return RET_OK;
}

InnerContext::~InnerContext() {
  if (this->thread_pool_ != nullptr) {
    DestroyThreadPool(this->thread_pool_);
    free(this->thread_pool_);
    this->thread_pool_ = nullptr;
  }
}

int InnerContext::IsValid() const {
  if (this->device_list_.empty()) {
    MS_LOG(ERROR) << "Device list is empty.";
    return RET_NOT_SUPPORT;
  }
  if (!IsCpuEnabled()) {
    MS_LOG(ERROR) << "CPU is not supported.";
    return RET_NOT_SUPPORT;
  }
#ifndef SUPPORT_GPU
  if (IsGpuEnabled()) {
    MS_LOG(ERROR) << "GPU is not supported.";
    return RET_NOT_SUPPORT;
  }
#endif
#ifndef SUPPORT_NPU
  if (IsNpuEnabled()) {
    MS_LOG(ERROR) << "NPU is not supported.";
    return RET_NOT_SUPPORT;
  }
#endif
  return RET_OK;
}

bool InnerContext::IsCpuFloat16Enabled() const {
  if (!IsCpuEnabled()) {
    return false;
  }
  return GetCpuInfo().enable_float16_;
}

bool InnerContext::IsGpuFloat16Enabled() const {
  if (!IsGpuEnabled()) {
    return false;
  }
  return GetGpuInfo().enable_float16_;
}

bool InnerContext::IsCpuEnabled() const {
  return this->device_list_.end() !=
         std::find_if(this->device_list_.begin(), this->device_list_.end(),
                      [](const DeviceContext &device) { return device.device_type_ == DT_CPU; });
}

bool InnerContext::IsGpuEnabled() const {
  return this->device_list_.end() !=
         std::find_if(this->device_list_.begin(), this->device_list_.end(),
                      [](const DeviceContext &device) { return device.device_type_ == DT_GPU; });
}

bool InnerContext::IsNpuEnabled() const {
#ifdef SUPPORT_NPU
  MS_ASSERT(npu_manager_ != nullptr);
  return this->device_list_.end() !=
           std::find_if(this->device_list_.begin(), this->device_list_.end(),
                        [](const DeviceContext &device) { return device.device_type_ == DT_NPU; }) &&
         npu_manager_->IsSupportNPU();
#else
  return false;
#endif
}

CpuDeviceInfo InnerContext::GetCpuInfo() const {
  auto iter = std::find_if(this->device_list_.begin(), this->device_list_.end(),
                           [](const DeviceContext &device) { return device.device_type_ == DT_CPU; });
  if (iter == this->device_list_.end()) {
    return {};
  } else {
    return iter->device_info_.cpu_device_info_;
  }
}

GpuDeviceInfo InnerContext::GetGpuInfo() const {
  auto iter = std::find_if(this->device_list_.begin(), this->device_list_.end(),
                           [](const DeviceContext &device) { return device.device_type_ == DT_GPU; });
  if (iter == this->device_list_.end()) {
    return {};
  } else {
    return iter->device_info_.gpu_device_info_;
  }
}

NpuDeviceInfo InnerContext::GetNpuInfo() const {
  auto iter = std::find_if(this->device_list_.begin(), this->device_list_.end(),
                           [](const DeviceContext &device) { return device.device_type_ == DT_NPU; });
  if (iter == this->device_list_.end()) {
    return {};
  } else {
    return iter->device_info_.npu_device_info_;
  }
}

}  // namespace mindspore::lite
