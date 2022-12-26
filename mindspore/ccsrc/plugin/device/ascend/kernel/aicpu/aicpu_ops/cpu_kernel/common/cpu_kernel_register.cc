/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "common/cpu_kernel_register.h"

#include <mutex>
#include <vector>
#include <memory>

#include "aicpu_sharder/aicpu_context.h"
#include "aicpu_sharder/aicpu_async_event.h"
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/status.h"
#include "cpu_kernel/common/async_event_util.h"
#include "cpu_kernel/common/async_cpu_kernel.h"

namespace {
#define TYPE_REGISTAR(type, fun) type##Registerar(type, fun)
// protect creatorMap_
std::mutex g_mutex;
}  // namespace

namespace aicpu {
/*
 * register kernel.
 */
bool RegistCpuKernel(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Registerar TYPE_REGISTAR(type, fun);
  return true;
}

/*
 * get instance.
 * @return CpuKernelRegister &: CpuKernelRegister instance
 */
CpuKernelRegister &CpuKernelRegister::Instance() {
  static CpuKernelRegister instance;
  return instance;
}

/*
 * get cpu kernel.
 * param opType: the op type of kernel
 * @return shared_ptr<CpuKernel>: cpu kernel ptr
 */
std::shared_ptr<CpuKernel> CpuKernelRegister::GetCpuKernel(const std::string &opType) {
  std::unique_lock<std::mutex> lock(g_mutex);
  auto iter = creatorMap_.find(opType);
  if (iter != creatorMap_.end()) {
    return iter->second();
  }
  KERNEL_LOG_WARN("The kernel[%s] is not registered.", opType.c_str());
  return std::shared_ptr<CpuKernel>(nullptr);
}

/*
 * get all cpu kernel registered op types.
 * @return std::vector<string>: all cpu kernel registered op type
 */
std::vector<std::string> CpuKernelRegister::GetAllRegisteredOpTypes() const {
  std::vector<std::string> ret;
  std::unique_lock<std::mutex> lock(g_mutex);
  for (auto iter = creatorMap_.begin(); iter != creatorMap_.end(); ++iter) {
    ret.push_back(iter->first);
  }

  return ret;
}

/*
 * run cpu kernel.
 * param ctx: context of kernel
 * @return uint32_t: 0->success other->failed
 */
uint32_t CpuKernelRegister::RunCpuKernel(CpuKernelContext &ctx) {
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("RunCpuKernel[%s] begin.", type.c_str());
  auto kernel = GetCpuKernel(type);
  if (kernel == nullptr) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (aicpu::SetThreadLocalCtx != nullptr) {
    if (aicpu::SetThreadLocalCtx(aicpu::kContextKeyOpName, type) != aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set kernel name[%s] to context failed.", type.c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  if (aicpu::SetOpname != nullptr) {
    (void)aicpu::SetOpname(type);
  }

  auto start = std::chrono::steady_clock::now();
  uint32_t ret = kernel->Compute(ctx);
  auto end = std::chrono::steady_clock::now();
  double dr_us = std::chrono::duration<double, std::micro>(end - start).count();
  KERNEL_LOG_EVENT("RunCpuKernel[%s], run time is [%lf] us.", type.c_str(), dr_us);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  KERNEL_LOG_INFO("RunCpuKernel[%s] success.", type.c_str());
  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelRegister::RunCpuKernelAsync(CpuKernelContext &ctx, const uint8_t wait_type, const uint32_t wait_id,
                                              std::function<uint32_t()> cb) {
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("RunCpuKernelAsync[%s] begin.", type.c_str());
  auto kernel = GetCpuKernel(type);
  if (kernel == nullptr) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  AsyncCpuKernel *async_kernel = dynamic_cast<AsyncCpuKernel *>(kernel.get());
  if (async_kernel == nullptr) {
    KERNEL_LOG_ERROR("kernel name[%s] does not hava async impl.", type.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (aicpu::SetThreadLocalCtx != nullptr) {
    if (aicpu::SetThreadLocalCtx(aicpu::kContextKeyOpName, type) != aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set kernel name[%s] to context failed.", type.c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
    if (aicpu::SetThreadLocalCtx(aicpu::kContextKeyWaitType, std::to_string(wait_type)) != aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set wait type to context failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
    if (aicpu::SetThreadLocalCtx(aicpu::kContextKeyWaitId, std::to_string(wait_id)) != aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set wait id to context failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  if (aicpu::SetOpname != nullptr) {
    (void)aicpu::SetOpname(type);
  }
  std::shared_ptr<AsyncNotifyInfo> notify_info = std::make_shared<AsyncNotifyInfo>();
  aicpu::GetTaskAndStreamId(&notify_info->task_id, &notify_info->stream_id);
  (void)aicpu::aicpuGetContext(&notify_info->ctx);
  notify_info->wait_type = wait_type;
  notify_info->wait_id = wait_id;

  auto start = std::chrono::steady_clock::now();
  auto done = [notify_info, kernel, type, cb, start](uint32_t status) {
    auto end = std::chrono::steady_clock::now();
    double dr_us = std::chrono::duration<double, std::micro>(end - start).count();
    KERNEL_LOG_EVENT("RunCpuKernel[%s], run time is [%lf] us.", type.c_str(), dr_us);
    if (status == KERNEL_STATUS_OK) {
      KERNEL_LOG_INFO("RunCpuKernel[%s] success.", type.c_str());
      status = cb();
    }
    notify_info->ret_code = status;
    void *param = reinterpret_cast<void *>(notify_info.get());
    KERNEL_LOG_INFO(
      "RunCpuKernelAsync notify event wait, wait_type[%u], "
      "wait_id[%u], task_id[%u], stream_id[%u], status[%u].",
      notify_info->wait_type, notify_info->wait_id, notify_info->task_id, notify_info->stream_id,
      notify_info->ret_code);
    AsyncEventUtil::GetInstance().NotifyWait(param, sizeof(AsyncNotifyInfo));
  };
  return async_kernel->ComputeAsync(ctx, done);
}

CpuKernelRegister::Registerar::Registerar(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Instance().Register(type, fun);
}

// register creator, this function will call in the constructor
void CpuKernelRegister::Register(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
  std::unique_lock<std::mutex> lock(g_mutex);
  std::map<std::string, KERNEL_CREATOR_FUN>::iterator iter = creatorMap_.find(type);
  if (iter != creatorMap_.end()) {
    KERNEL_LOG_WARN("Register[%s] creator already exist", type.c_str());
    return;
  }

  creatorMap_[type] = fun;
  KERNEL_LOG_DEBUG("Kernel[%s] register successfully", type.c_str());
}
}  // namespace aicpu
