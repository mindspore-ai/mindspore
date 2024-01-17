/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_synchronizer.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#include "plugin/device/cpu/hal/device/cpu_device_synchronizer.h"
#include "include/transform/graph_ir/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
::ge::MemBlock *GeAllocator::Malloc(size_t size) {
  auto addr = res_manager_->AllocateMemory(size);
  MS_LOG(DEBUG) << "GE Allocator malloc addr: " << addr << " size: " << size;
  auto mem_block = new ::ge::MemBlock(*this, addr, size);
  return mem_block;
}

void GeAllocator::Free(::ge::MemBlock *block) {
  res_manager_->FreeMemory(block->GetAddr());
  MS_LOG(DEBUG) << "GE Allocator free addr: " << block->GetAddr();
  delete block;
}

void GeDeviceResManager::Initialize() {
  if (IsEnableRefMode()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    runtime_instance_ = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    if (!runtime_instance_->Init()) {
      MS_LOG(EXCEPTION) << "Kernel runtime init error.";
    }
    mem_manager_ = runtime_instance_->GetMemoryManager();
  } else {
    mem_manager_ = std::make_shared<cpu::CPUMemoryManager>();
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
}

void GeDeviceResManager::Destroy() {
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}

bool GeDeviceResManager::AllocateMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
  if (IsEnableRefMode() && device_name_in_address != device_context_->device_context_key().device_name_) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                      << ", type name in context:" << device_context_->device_context_key().device_name_;
  }

  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  if (runtime_instance_ != nullptr) {
    runtime_instance_->SetContext();
  }
  auto size =
    address->type_id() == kObjectTypeString ? address->GetSize() + sizeof(ge::StringHead) : address->GetSize();
  void *device_ptr = mem_manager_->MallocMemFromMemPool(size, address->from_persistent_mem(), address->need_recycle());
  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  return true;
}

void *GeDeviceResManager::AllocateMemory(size_t size) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, false);
}

size_t GeDeviceResManager::GetMaxUsedMemorySize() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetMaxUsedMemorySize();
}

void GeDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

void GeDeviceResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                         const std::vector<size_t> &keep_addr_sizes) const {
  AscendMemoryPool::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

void GeDeviceResManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
}

void GeDeviceResManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
}

std::vector<void *> GeDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list) const {
  return mem_manager_->MallocContinuousMemFromMemPool(size_list);
}

DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (IsEnableRefMode()) {
    if (kernel_tensor->device_name().empty()) {
      kernel_tensor->set_device_name(device_context_->device_context_key().device_name_);
      kernel_tensor->set_device_id(device_context_->device_context_key().device_id_);
    }
    auto device_address = std::make_shared<AscendDeviceAddress>(kernel_tensor);
    device_address->set_device_synchronizer(std::make_shared<AscendDeviceSynchronizer>());
    return device_address;
  } else {
    if (kernel_tensor->device_name().empty()) {
      kernel_tensor->set_device_name(kCPUDevice);
      kernel_tensor->set_device_id(0);
    }
    auto device_address = std::make_shared<cpu::CPUDeviceAddress>(kernel_tensor);
    device_address->set_device_synchronizer(std::make_shared<cpu::CPUDeviceSynchronizer>());
    return device_address;
  }
}

void GeDeviceResManager::GeSetContextOptions(const std::shared_ptr<MsContext> &ms_context_ptr,
                                             transform::SessionOptions *options) {
  MS_EXCEPTION_IF_NULL(options);
  if (ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.graphMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE);
  }

  if (ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.variableMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
  }

  auto atomic_clean_policy = ms_context_ptr->get_param<std::string>(MS_CTX_ATOMIC_CLEAN_POLICY);
  if (atomic_clean_policy.empty()) {
    atomic_clean_policy = "1";
  }
  (*options)["ge.exec.atomicCleanPolicy"] = atomic_clean_policy;
  MS_LOG(INFO) << "Set GE atomic clean policy to " << atomic_clean_policy << ".";
  (*options)["ge.graphRunMode"] = "1";
}

void GeDeviceResManager::CreateSessionAndGraphRunner() {
  std::shared_ptr<::ge::Session> sess = transform::GetGeSession();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (sess == nullptr) {
    transform::SessionOptions options;
    options["ge.enablePrintOpPass"] = "0";
    GeSetContextOptions(ms_context, &options);
    options["ge.constLifecycle"] = "graph";

    options["ge.exec.formatMode"] = "0";
    if (common::GetEnv("MS_FORMAT_MODE") == "1") {
      options["ge.exec.formatMode"] = "1";
    }

    SetPassthroughGeOptions(false, &options);

    sess = transform::NewSession(options);
    transform::SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  if (ms_context->EnableAoeOnline()) {
    transform::DfGraphManager::GetInstance().AoeGeGraph();
  }
  auto graph_runner = transform::NewGraphRunner(options);
  transform::SetGraphRunner(graph_runner);
}

bool GeDeviceResManager::BindDeviceToCurrentThread(bool force_bind) const {
  if (runtime_instance_ != nullptr) {
    if (force_bind) {
      runtime_instance_->SetContextForce();
    } else {
      runtime_instance_->SetContext();
    }
  }
  return true;
}

void GeDeviceResManager::ResetStreamAndCtx() {
  if (runtime_instance_ != nullptr) {
    runtime_instance_->ResetStreamAndCtx();
  }
}

bool GeDeviceResManager::CreateStream(size_t *stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStream(stream_id);
  return true;
}

bool GeDeviceResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStreamWithFlags(stream_id, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC,
                                                       IntToUint(priority));
  return true;
}

bool GeDeviceResManager::single_op_multi_stream_enable() const {
  return AscendStreamMng::GetInstance().single_op_multi_stream_enable();
}

void GeDeviceResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return AscendStreamMng::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *GeDeviceResManager::GetStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return nullptr;
  }
  return AscendStreamMng::GetInstance().GetStream(stream_id);
}

void GeDeviceResManager::SetCurrentStreamId(size_t stream_id) {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return;
  }
  AscendStreamMng::GetInstance().set_current_stream(stream_id);
}

size_t GeDeviceResManager::GetCurrentStreamId() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().current_stream();
}

bool GeDeviceResManager::QueryStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().QueryStream(stream_id);
}

bool GeDeviceResManager::SyncStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncStream(stream_id);
}

bool GeDeviceResManager::SyncAllStreams() const {
  if (runtime_instance_ == nullptr) {
    return true;
  }
  runtime_instance_->SetContext();
  return AscendStreamMng::GetInstance().SyncAllStreams();
}

bool GeDeviceResManager::SyncNotDefaultStreams() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncNotDefaultStreams();
}

size_t GeDeviceResManager::DefaultStream() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().default_stream_id();
}

DeviceEventPtr GeDeviceResManager::CreateEventWithFlag(bool enable_timing, bool blocking) const {
  auto flag = enable_timing ? ACL_EVENT_TIME_LINE : ACL_EVENT_DEFAULT;
  auto event = std::make_shared<AscendEvent>(flag);
  MS_EXCEPTION_IF_NULL(event);
  return event;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
