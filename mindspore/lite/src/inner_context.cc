/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <memory>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#ifdef ENABLE_MINDRT
#include "thread/actor_threadpool.h"
#endif
#ifdef SUPPORT_NPU
#include "include/HiAiModelManagerType.h"
#endif
#ifdef GPU_OPENCL
#include "src/runtime/gpu/opencl/opencl_runtime.h"
#endif

namespace mindspore::lite {
namespace {
const constexpr int kMaxLiteContextDeviceNums = 2;
const constexpr int kMaxInnerContextDeviceNums = 3;
}  // namespace

void InnerContext::InitDeviceFp16() {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  CpuInfo cpu_info;
  device_and_pkg_support_fp16_ = cpu_info.ArmIsSupportFp16();
#else
  device_and_pkg_support_fp16_ = false;
#endif
}

InnerContext::InnerContext(const Context *context) {
  if (context != nullptr) {
    this->allocator = context->allocator;
    this->thread_num_ = context->thread_num_;
    this->enable_parallel_ = context->enable_parallel_;
    this->affinity_core_list_ = context->affinity_core_list_;
    SetContextDevice(context);
    this->delegate = context->delegate;
    this->float_mode = context->float_mode;
  }
  InitDeviceFp16();
}

void InnerContext::SetContextDevice(const Context *context) {
  this->device_list_.clear();

  if (context->device_list_.size() > kMaxLiteContextDeviceNums || context->device_list_.size() <= 0) {
    return;
  }
  if (context->device_list_.front().device_type_ != DT_CPU) {
    return;
  }

  /* user set order for different device */
  if (context->device_list_.size() < kMaxLiteContextDeviceNums) {
    this->device_list_.push_back(context->device_list_.front());
    return;
  }

  /* keep compatibility :
   * if user set CPU & NPU/GPU
   * NPU/GPU higher priority */
  bool isUserSetNPU = context->device_list_.end() !=
                      std::find_if(this->device_list_.begin(), this->device_list_.end(),
                                   [](const DeviceContext &device) { return device.device_type_ == DT_NPU; });
  bool isUserSetGPU = context->device_list_.end() !=
                      std::find_if(this->device_list_.begin(), this->device_list_.end(),
                                   [](const DeviceContext &device) { return device.device_type_ == DT_GPU; });
  if (isUserSetGPU == false && isUserSetNPU == false) {
    return;
  }

  /* add GPU/NPU first */
  for (auto &device_ctx : context->device_list_) {
    if (device_ctx.device_type_ != DT_CPU) {
      this->device_list_.push_back(device_ctx);
    }
  }

  /* add CPU */
  for (auto &device_ctx : context->device_list_) {
    if (device_ctx.device_type_ == DT_CPU) {
      if (isUserSetNPU || (isUserSetGPU && enable_parallel_ == false)) {
        auto cpu_ctx = device_ctx;
        cpu_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = NO_BIND;
        this->device_list_.push_back(cpu_ctx);
      } else {
        this->device_list_.push_back(device_ctx);
      }
    }
  }
  return;
}

int InnerContext::CreateThreadPool() {
  if (this->thread_pool_ == nullptr) {
    BindMode bind_mode = Power_NoBind;
    if (this->IsCpuEnabled()) {
      bind_mode = static_cast<BindMode>(this->GetCpuDeviceInfo()->cpu_bind_mode_);
    }

#ifdef ENABLE_MINDRT
#ifdef OPERATOR_PARALLELISM
    int actor_parallel_thread = this->enable_parallel_ ? (this->thread_num_ > 2 ? (this->thread_num_ / 2) : 1) : 1;
#else
    int actor_parallel_thread = this->enable_parallel_ ? kDefaultParallelNum : 1;
#endif
    if (this->affinity_core_list_.empty()) {
      thread_pool_ = ActorThreadPool::CreateThreadPool(actor_parallel_thread, this->thread_num_, bind_mode);
      MS_CHECK_TRUE_MSG(thread_pool_ != nullptr, RET_NULL_PTR, "Create Allocator failed");
    } else {
      thread_pool_ =
        ActorThreadPool::CreateThreadPool(actor_parallel_thread, this->thread_num_, this->affinity_core_list_);
      MS_CHECK_TRUE_MSG(thread_pool_ != nullptr, RET_NULL_PTR, "Create Allocator failed");
    }
#else
    thread_pool_ = ThreadPool::CreateThreadPool(thread_num_ - 1);
    thread_pool_->SetCpuAffinity(static_cast<mindspore::BindMode>(bind_mode));
#endif
  }
  return RET_OK;
}
int InnerContext::Init() {
  if (this->IsValid() != RET_OK) {
    MS_LOG(ERROR) << "Context is not valid";
    return RET_NOT_SUPPORT;
  }

  if (CreateThreadPool()) {
    MS_LOG(ERROR) << "CreateThreadPool failed.";
    return RET_ERROR;
  }

  if (this->allocator == nullptr) {
#ifdef SERVER_INFERENCE
    this->allocator = std::make_shared<DynamicMemAllocator>(node_id_);
#else
    this->allocator = mindspore::Allocator::Create();
#endif
    CHECK_NULL_RETURN(this->allocator);
  }
  if (IsNpuEnabled()) {
    MS_LOG(DEBUG) << "NPU enabled.";
#ifdef SUPPORT_NPU
    for (auto &device_ctx : this->device_list_) {
      if (device_ctx.device_type_ == DT_NPU &&
          device_ctx.device_info_.npu_device_info_.frequency_ != hiai::AiModelDescription_Frequency_LOW &&
          device_ctx.device_info_.npu_device_info_.frequency_ != hiai::AiModelDescription_Frequency_MEDIUM &&
          device_ctx.device_info_.npu_device_info_.frequency_ != hiai::AiModelDescription_Frequency_HIGH &&
          device_ctx.device_info_.npu_device_info_.frequency_ != hiai::AiModelDescription_Frequency_EXTREME) {
        MS_LOG(WARNING) << "NPU frequency set to 3, original value "
                        << device_ctx.device_info_.npu_device_info_.frequency_;
        device_ctx.device_info_.npu_device_info_.frequency_ = hiai::AiModelDescription_Frequency_HIGH;
      }
    }
#endif
  }
  if (IsGpuEnabled()) {
    MS_LOG(DEBUG) << "GPU enabled.";
  }
  return RET_OK;
}

InnerContext::~InnerContext() {
  if (this->thread_pool_ != nullptr) {
    delete thread_pool_;
    this->thread_pool_ = nullptr;
  }
}

int InnerContext::IsValid() const {
  if (this->device_list_.empty()) {
    MS_LOG(ERROR) << "Device list is empty.";
    return RET_NOT_SUPPORT;
  }
  if (this->device_list_.size() > kMaxInnerContextDeviceNums) {
    MS_LOG(ERROR) << "Not support device list more than " << kMaxInnerContextDeviceNums;
    return RET_NOT_SUPPORT;
  }
  if (thread_num_ < 1) {
    MS_LOG(ERROR) << "Thread num smaller than 1 is not allowed.";
    return RET_NOT_SUPPORT;
  }
  if (!IsAllDeviceTypeValid()) {
    MS_LOG(ERROR) << "Device type should be one of DT_CPU, DT_GPU or DT_NPU.";
    return RET_NOT_SUPPORT;
  }

  if (IsCpuBindModeInvalid()) {
    MS_LOG(ERROR) << "CPU bind mode should be one of NO_BIND, HIGHER_CPU or MID_CPU.";
    return RET_NOT_SUPPORT;
  }

#ifndef SUPPORT_GPU
  if (IsUserSetGpu()) {
    MS_LOG(ERROR) << "GPU is not supported.";
    return RET_NOT_SUPPORT;
  }
#endif
#ifndef SUPPORT_NPU
  if (IsUserSetNpu()) {
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
  if (!device_and_pkg_support_fp16_) {
    return false;
  }
  return GetCpuInfo().enable_float16_;
}

bool InnerContext::IsGpuFloat16Enabled() const {
#ifdef GPU_OPENCL
  if (!IsGpuEnabled()) {
    return false;
  }
  opencl::OpenCLRuntimeInnerWrapper wrapper;
  if (!wrapper.GetInstance()->GetFp16Enable()) {
    return false;
  }
  return GetGpuInfo().enable_float16_;
#else
  return false;
#endif
}

#ifdef ENABLE_OPENGL_TEXTURE
bool InnerContext::IsGLTextureEnabled() const { return GetGpuInfo().enable_gl_texture_; }
#endif

bool InnerContext::IsCpuEnabled() const { return IsUserSetCpu(); }

const CpuDeviceInfo *InnerContext::GetCpuDeviceInfo() const {
  if (IsUserSetCpu() == false) {
    return nullptr;
  }
  const DeviceInfo *device_info = nullptr;

  (void)std::find_if(this->device_list_.begin(), this->device_list_.end(), [&](const DeviceContext &device) {
    if (device.device_type_ == DeviceType::DT_CPU) {
      device_info = &device.device_info_;
      return true;
    }
    return false;
  });

  return reinterpret_cast<const CpuDeviceInfo *>(device_info);
}

bool InnerContext::IsGpuEnabled() const {
#ifdef SUPPORT_GPU
  return IsUserSetGpu();
#else
  return false;
#endif
}

bool InnerContext::IsNpuEnabled() const {
#ifdef SUPPORT_NPU
  return IsUserSetNpu();
#else
  return false;
#endif
}

bool InnerContext::IsProviderEnabled() const {
  return this->device_list_.end() !=
         std::find_if(this->device_list_.begin(), this->device_list_.end(),
                      [](const DeviceContext &device) { return !device.provider_.empty(); });
}

bool InnerContext::IsAllDeviceTypeValid() const {
  return std::all_of(this->device_list_.begin(), this->device_list_.end(), [](const DeviceContext &device) {
    return device.device_type_ >= DT_CPU && device.device_type_ < DT_END;
  });
}

bool InnerContext::IsCpuBindModeInvalid() const {
  return this->device_list_.end() !=
         std::find_if(this->device_list_.begin(), this->device_list_.end(), [](const DeviceContext &device) {
           return device.device_type_ == DT_CPU && (device.device_info_.cpu_device_info_.cpu_bind_mode_ < NO_BIND ||
                                                    device.device_info_.cpu_device_info_.cpu_bind_mode_ > MID_CPU);
         });
}

bool InnerContext::IsUserSetCpu() const {
  return this->device_list_.end() !=
         std::find_if(this->device_list_.begin(), this->device_list_.end(),
                      [](const DeviceContext &device) { return device.device_type_ == DT_CPU; });
}

bool InnerContext::IsUserSetGpu() const {
  return this->device_list_.end() !=
         std::find_if(this->device_list_.begin(), this->device_list_.end(),
                      [](const DeviceContext &device) { return device.device_type_ == DT_GPU; });
}

bool InnerContext::IsUserSetNpu() const {
  return this->device_list_.end() !=
         std::find_if(this->device_list_.begin(), this->device_list_.end(),
                      [](const DeviceContext &device) { return device.device_type_ == DT_NPU; });
}

std::set<std::string> InnerContext::GetProviders() const {
  std::set<std::string> providers;
  for (auto &&device : device_list_) {
    if (!device.provider_.empty()) {
      providers.insert(device.provider_);
    }
  }
  return providers;
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

ThreadPool *InnerContext::thread_pool() const { return thread_pool_; }

bool InnerContext::device_and_pkg_support_fp16() const { return this->device_and_pkg_support_fp16_; }

std::set<void *> InnerContext::GetLinkInfo(void *pre) const {
  auto iter = link_info_.find(pre);
  if (iter == link_info_.end()) {
    MS_LOG(DEBUG) << "Not found precursor in link information.";
    return {};
  }
  return iter->second;
}

std::unordered_map<void *, std::set<void *>> InnerContext::GetAllLinkInfo() const { return link_info_; }

void InnerContext::SetLinkInfo(void *pre, void *suc) {
  auto iter = link_info_.find(pre);
  if (iter != link_info_.end()) {
    iter->second.insert(suc);
    return;
  }
  std::set<void *> suc_set{suc};
  link_info_[pre] = suc_set;
}

void InnerContext::SetAllLinkInfo(const std::unordered_map<void *, std::set<void *>> &all_link_info) {
  link_info_ = all_link_info;
}

void InnerContext::ReplaceLinkInfoReceiverWithNewOne(void *new_receiver, void *old_receiver) {
  for (auto &info : link_info_) {
    auto &receivers = info.second;
    auto iter = receivers.find(old_receiver);
    if (iter != receivers.end()) {
      receivers.erase(iter);
      receivers.insert(new_receiver);
    }
  }
}

void InnerContext::ReplaceLinkInfoSenderWithNewOne(void *new_sender, void *old_sender) {
  auto receiver_set = this->GetLinkInfo(old_sender);
  for (auto item : receiver_set) {
    this->SetLinkInfo(new_sender, item);
  }
}

int ParallelLaunch(const Context *context, const Func &func, Content content, int task_num) {
  ThreadPool *pool = static_cast<const lite::InnerContext *>(context)->thread_pool();
  if (pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return RET_NULL_PTR;
  }
  return pool->ParallelLaunch(func, content, task_num);
}
}  // namespace mindspore::lite
