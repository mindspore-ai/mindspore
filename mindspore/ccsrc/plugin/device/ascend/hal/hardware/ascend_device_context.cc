/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_device_context.h"
#include <memory>
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#include "debug/debugger/proto_exporter.h"
#endif
#include "include/backend/debug/debugger/proto_exporter.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "include/backend/distributed/collective/collective_manager.h"

using mindspore::profiler::ascend::AscendProfiler;
#endif

namespace mindspore {
namespace device {
namespace ascend {
void AscendDeviceContext::Initialize() {
  if (initialized_) {
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    runtime_instance_->SetContext();
    return;
  } else {
    MS_LOG(INFO) << "Start Initialize...";
#ifndef ENABLE_SECURITY
    AscendProfiler::GetInstance()->MsprofInitProfiler();
#endif
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = dynamic_cast<AscendKernelRuntime *>(
    device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id));
  MS_EXCEPTION_IF_NULL(runtime_instance_);
#ifndef ENABLE_SECURITY
  runtime_instance_->PreInit();
#endif
  // OpenTsd when enable hccl, keep consistent with before
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    MS_EXCEPTION_IF_NULL(GetDeprecatedInterface());
    GetDeprecatedInterface()->OpenTsd(MsContext::GetInstance());
  }
  runtime_instance_->SetRtDevice(device_id);

  // enable hccl and init hccl not done, skip the rest step.
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) &&
      !distributed::collective::CollectiveManager::instance()->initialized()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();
  auto ascend_res_manager = dynamic_cast<AscendDeviceResManager *>(device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(ascend_res_manager);
  runtime_instance_ = ascend_res_manager->runtime_instance_;
  auto ascend_kernel_executor = dynamic_cast<AscendKernelExecutor *>(kernel_executor_.get());
  MS_EXCEPTION_IF_NULL(ascend_kernel_executor);
  ascend_kernel_executor->Initialize();
  auto ascend_graph_executor = dynamic_cast<AscendGraphExecutor *>(graph_executor_.get());
  MS_EXCEPTION_IF_NULL(ascend_graph_executor);
  ascend_graph_executor->Initialize();
  initialized_ = true;
  MS_LOG(INFO) << "Initialize success.";
}

void AscendDeviceContext::Destroy() {
#ifndef ENABLE_SECURITY
  AscendProfiler::GetInstance()->MsprofStopProfiler();
#endif
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger && debugger->debugger_enabled()) {
    debugger->SetTrainingDone(true);
    bool ret = debugger->SendMetadata(false);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to SendMetadata when finalize";
    }
  }
#endif
  MS_LOG(INFO) << "Enter Destroy...";
  if (!initialized_) {
    if (deprecated_interface_ != nullptr) {
      (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
    }
    return;
  }

  MS_LOG(INFO) << "Start Destroy ";
  auto ascend_graph_executor = dynamic_cast<AscendGraphExecutor *>(graph_executor_.get());
  ascend_graph_executor->Destroy();
  auto ascend_kernel_executor = dynamic_cast<AscendKernelExecutor *>(kernel_executor_.get());
  ascend_kernel_executor->Destroy();
  device_res_manager_->Destroy();
  if (runtime_instance_) {
    runtime_instance_ = nullptr;
  }
  if (deprecated_interface_ != nullptr) {
    (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
  }
  initialized_ = false;
  MS_LOG(INFO) << "Destroy success.";
}

// @todo move SetRunMode to here when old runtime is delete
bool AscendDeviceContext::PartitionGraph(const FuncGraphPtr &func_graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
}

RunMode AscendDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) && !IsDynamicShapeGraph(func_graph)) {
    return RunMode::kGraphMode;
  } else {
    return RunMode::kKernelMode;
  }
}

DeprecatedInterface *AscendDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>(nullptr);
  }
  return deprecated_interface_.get();
}

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
MS_REGISTER_DEVICE(kDavinciMultiGraphInferenceDevice, AscendDeviceContext);
#ifdef WITH_BACKEND
MSCONTEXT_REGISTER_INIT_FUNC(kAscendDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  auto enable_ge = mindspore::common::GetEnv("MS_ENABLE_GE");
  if (enable_ge == "1") {
    if (ctx->backend_policy() != "ge") {
      ctx->set_backend_policy("ge");
    }
  } else {
    if (ctx->backend_policy() != "ms") {
      ctx->set_backend_policy("ms");
    }
  }
});
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
