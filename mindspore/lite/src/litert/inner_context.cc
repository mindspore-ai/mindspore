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
#include "src/litert/inner_context.h"
#include <algorithm>
#include <memory>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#ifdef ENABLE_MINDRT
#include "thread/actor_threadpool.h"
#include "thread/parallel_threadpool.h"
#endif
#ifdef SUPPORT_NPU
#include "include/HiAiModelManagerType.h"
#endif
#ifdef GPU_OPENCL
#include "src/litert/kernel/gpu/opencl/opencl_runtime.h"
#endif
#include "src/litert/inner_allocator.h"
#include "experimental/src/exec_env_utils.h"
#include "src/litert/thread_pool_reuse_manager.h"

namespace mindspore::lite {
namespace {
const constexpr int kMaxInnerContextDeviceNums = 3;
const constexpr int kNumCoreNumTimes = 5;
constexpr int kDefaultParallelNum = 2;
}  // namespace

InnerContext::InnerContext() {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  CpuInfo cpu_info;
  device_and_pkg_support_fp16_ = cpu_info.ArmIsSupportFp16();
#endif
}

void InnerContext::InitExperimentalExecEnv() {
#ifdef MSLITE_ENABLE_EXPERIMENTAL_KERNEL
  exec_env_.allocator = this->allocator.get();
  exec_env_.threadPool = this->thread_pool_;
  exec_env_.alloc = experimental::DefaultAllocatorMalloc;
  exec_env_.free = experimental::DefaultAllocatorFree;
  exec_env_.parallelLaunch = experimental::DefaultThreadPoolParallelLunch;
#endif
}

int InnerContext::CreateThreadPool() {
  if (this->thread_pool_ == nullptr) {
    bind_mode_ = Power_NoBind;
    if (this->IsDeviceTypeEnabled(DT_CPU)) {
      bind_mode_ = static_cast<BindMode>(this->GetDeviceInfo(DT_CPU).cpu_device_info_.cpu_bind_mode_);
    }
    this->inter_op_parallel_num_ =
      (!this->enable_parallel_ && this->inter_op_parallel_num_ > 1) ? this->inter_op_parallel_num_ : 1;
    actor_thread_num_ = (inter_op_parallel_num_ > 1) ? 1 : (this->enable_parallel_ ? kDefaultParallelNum : 1);
    thread_pool_ = ThreadPoolReuseManager::GetInstance()->GetThreadPool(
      actor_thread_num_, inter_op_parallel_num_, thread_num_, bind_mode_, affinity_core_list_, runner_id_);
    if (thread_pool_ == nullptr) {
#ifdef ENABLE_MINDRT
      if (inter_op_parallel_num_ > 1) {
        thread_pool_ = ParallelThreadPool::CreateThreadPool(this->inter_op_parallel_num_, this->thread_num_,
                                                            this->affinity_core_list_, bind_mode_, runner_id_);
      } else {
        thread_pool_ = ActorThreadPool::CreateThreadPool(actor_thread_num_, this->thread_num_,
                                                         this->affinity_core_list_, bind_mode_);
      }
#else
      thread_pool_ = ThreadPool::CreateThreadPool(thread_num_ - 1);
      thread_pool_->SetCpuAffinity(static_cast<mindspore::BindMode>(bind_mode_));
#endif
    }
  }
  MS_CHECK_TRUE_MSG(thread_pool_ != nullptr, RET_NULL_PTR, "Create Allocator failed");
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
    this->allocator = mindspore::Allocator::Create();
    CHECK_NULL_RETURN(this->allocator);
  }
  if (IsDeviceTypeEnabled(DT_NPU)) {
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
  InitExperimentalExecEnv();
  return RET_OK;
}

InnerContext::~InnerContext() {
  MS_LOG(INFO) << "delete InnerContext.";
  ThreadPoolReuseManager::GetInstance()->RetrieveThreadPool(actor_thread_num_, inter_op_parallel_num_, thread_num_,
                                                            bind_mode_, affinity_core_list_, thread_pool_);
  thread_pool_ = nullptr;
  MS_LOG(INFO) << "delete InnerContext done.";
}

int InnerContext::IsValid() {
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
  int core_num = static_cast<int>(std::max<size_t>(1, std::thread::hardware_concurrency()));
  int Threshold_thread_num = kNumCoreNumTimes * core_num;
  if (thread_num_ > Threshold_thread_num) {
    MS_LOG(WARNING) << "Thread num: " << thread_num_ << " is more than 5 times core num: " << Threshold_thread_num
                    << ", change it to 5 times core num. Please check whether Thread num is reasonable.";
    thread_num_ = Threshold_thread_num;
  }

  if (inter_op_parallel_num_ < 1) {
    MS_LOG(ERROR) << "InterOpParallelNum smaller than 1 is not allowed.";
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
  if (IsDeviceTypeEnabled(DT_GPU)) {
    MS_LOG(ERROR) << "GPU is not supported.";
    return RET_NOT_SUPPORT;
  }
#endif
#if !defined(SUPPORT_NPU) && !defined(SUPPORT_NNAPI)
  if (IsDeviceTypeEnabled(DT_NPU)) {
    MS_LOG(ERROR) << "NPU is not supported.";
    return RET_NOT_SUPPORT;
  }
#endif
#ifdef DELEGATE_CLIP
  if (this->delegate != nullptr) {
    MS_LOG(ERROR) << unsupport_delegate_log;
    return RET_NOT_SUPPORT;
  }
#endif
  return RET_OK;
}

bool InnerContext::IsCpuFloat16Enabled() const {
  if (!IsDeviceTypeEnabled(DT_CPU)) {
    return false;
  }
  if (!device_and_pkg_support_fp16_) {
    return false;
  }
  return GetDeviceInfo(DT_CPU).cpu_device_info_.enable_float16_;
}

bool InnerContext::IsGpuFloat16Enabled() const {
#ifdef GPU_OPENCL
  if (!IsDeviceTypeEnabled(DT_GPU)) {
    return false;
  }
  opencl::OpenCLRuntimeInnerWrapper wrapper;
  if (!wrapper.GetInstance()->GetFp16Enable()) {
    return false;
  }
  return GetDeviceInfo(DT_GPU).gpu_device_info_.enable_float16_;
#else
  return false;
#endif
}

bool InnerContext::IsNpuFloat16Enabled() const {
  if (!IsDeviceTypeEnabled(DT_NPU)) {
    return false;
  }
  if (!device_and_pkg_support_fp16_) {
    return false;
  }
  return GetDeviceInfo(DT_NPU).npu_device_info_.enable_float16_;
}

bool InnerContext::IsGLTextureEnabled() const {
#ifdef GPU_OPENCL
  if (!IsDeviceTypeEnabled(DT_GPU)) {
    return false;
  }
  return GetDeviceInfo(DT_GPU).gpu_device_info_.enable_gl_texture_;
#else
  return false;
#endif
}

bool InnerContext::IsDeviceTypeEnabled(DeviceType type) const {
  return device_list_.end() !=
         std::find_if(device_list_.begin(), device_list_.end(),
                      [type](const DeviceContext &device) { return device.device_type_ == type; });
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

int InnerContext::GetDelegateMode() const { return delegate_mode_; }

std::set<std::string> InnerContext::GetProviders() const {
  std::set<std::string> providers;
  for (auto &&device : device_list_) {
    if (!device.provider_.empty()) {
      providers.insert(device.provider_);
    }
  }
  return providers;
}

DeviceInfo InnerContext::GetDeviceInfo(DeviceType type) const {
  auto iter = std::find_if(device_list_.begin(), device_list_.end(),
                           [type](const DeviceContext &device) { return device.device_type_ == type; });
  if (iter == device_list_.end()) {
    return {};
  } else {
    return iter->device_info_;
  }
}

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
    (void)iter->second.insert(suc);
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
      (void)receivers.erase(iter);
      (void)receivers.insert(new_receiver);
    }
  }
}

void InnerContext::ReplaceLinkInfoSenderWithNewOne(void *new_sender, void *old_sender) {
  auto receiver_set = this->GetLinkInfo(old_sender);
  for (auto item : receiver_set) {
    this->SetLinkInfo(new_sender, item);
  }
}

int ParallelLaunch(const InnerContext *context, const Func &func, Content content, int task_num) {
  ThreadPool *pool = context->thread_pool_;
  if (pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return RET_NULL_PTR;
  }
  return pool->ParallelLaunch(func, content, task_num);
}
}  // namespace mindspore::lite
