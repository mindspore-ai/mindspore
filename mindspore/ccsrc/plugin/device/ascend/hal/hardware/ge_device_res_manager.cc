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
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_synchronizer.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#include "plugin/device/ascend/hal/device/ascend_pin_mem_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_synchronizer.h"
#include "include/transform/graph_ir/utils.h"
#include "graph/types.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"
#include "transform/acl_ir/op_api_util.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "graph/def_types.h"
#include "runtime/device/move_to.h"

namespace mindspore {
namespace device {
namespace ascend {
std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string cur_so_path = dl_info.dli_fname;
  return dirname(cur_so_path.data());
#else
  return "";
#endif
}

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
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  if (!runtime_instance_->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  mem_manager_ = runtime_instance_->GetMemoryManager();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    swap_manager_ = std::make_shared<SwapManager>(kDefaultStreamIndex, &AscendMemoryPool::GetInstance(),
                                                  &AscendPinMemPool::GetInstance());
  }
}

void GeDeviceResManager::SetCPUMemManager() {
  if (is_use_cpu_memory_) {
    return;
  }
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
  runtime_instance_ = nullptr;
  mem_manager_ = std::make_shared<cpu::CPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  is_use_cpu_memory_ = true;
}

void GeDeviceResManager::Destroy() {
  (void)DestroyAllEvents();
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}

bool GeDeviceResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
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
  void *device_ptr = nullptr;

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }

  if (swap_manager_ != nullptr) {
    device_ptr = swap_manager_->AllocDeviceMemory(address->GetSize(), stream_id);
  } else {
    device_ptr = mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem(),
                                                    address->need_recycle(), stream_id);
  }

  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  return true;
}

void *GeDeviceResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceMemory(size, stream_id);
  }
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
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

// Relevant function to manage memory statistics
size_t GeDeviceResManager::GetTotalMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalMemStatistics();
}

size_t GeDeviceResManager::GetTotalUsedMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalUsedMemStatistics();
}

size_t GeDeviceResManager::GetTotalIdleMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalIdleMemStatistics();
}

size_t GeDeviceResManager::GetTotalEagerFreeMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalEagerFreeMemStatistics();
}

size_t GeDeviceResManager::GetUsedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetUsedMemPeakStatistics();
}

size_t GeDeviceResManager::GetReservedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetReservedMemPeakStatistics();
}

std::unordered_map<std::string, std::size_t> GeDeviceResManager::GetBlockCountsStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> GeDeviceResManager::GetBlockUnitSizeStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockUnitSizeStatistics();
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
GeDeviceResManager::GetCommonMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetCommonMemBlocksInfoStatistics();
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
GeDeviceResManager::GetPersistentMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetPersistentMemBlocksInfoStatistics();
}

void GeDeviceResManager::ResetMaxMemoryReserved() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetMaxMemoryReserved();
}

void GeDeviceResManager::ResetMaxMemoryAllocated() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetMaxMemoryAllocated();
}

void GeDeviceResManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
}

void GeDeviceResManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
}

std::vector<void *> GeDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                                 uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> aligned_size_list;
  for (auto size : size_list) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
    aligned_size_list.emplace_back(align_size);
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceContinuousMem(aligned_size_list, stream_id);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(aligned_size_list, stream_id);
}

DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (!is_use_cpu_memory_) {
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

DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                         const Format &format, TypeId type_id,
                                                         const std::string &device_name, uint32_t device_id,
                                                         uint32_t stream_id) const {
  if (!is_use_cpu_memory_) {
    return std::make_shared<AscendDeviceAddress>(ptr, size, shape_vector, format, type_id, device_name, device_id,
                                                 stream_id);
  } else {
    return std::make_shared<cpu::CPUDeviceAddress>(ptr, size, shape_vector, format, type_id, kCPUDevice, device_id,
                                                   stream_id);
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
    auto format_mode = common::GetEnv("MS_FORMAT_MODE");
    if (format_mode == "1" || (format_mode.empty() && ms_context->ascend_soc_version() != "ascend910")) {
      MS_LOG(INFO) << "Set GE option ge.exec.formatMode to 1.";
      options["ge.exec.formatMode"] = "1";
    }

    SetPassthroughGeOptions(false, &options);

    sess = transform::NewSession(options);
    transform::SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = transform::NewGraphRunner(options);
  transform::SetGraphRunner(graph_runner);
}

bool GeDeviceResManager::LoadCollectiveCommLib() {
  // If this is simulation, load dummy collective communication library.
  if (!common::GetEnv(kSimulationLevel).empty()) {
    collective_comm_lib_ = &DummyAscendCollectiveCommLib::GetInstance();
    return true;
  }
  // Ascend backend supports HCCL and LCCL collective communication libraries.
  if (!common::GetEnv("MS_ENABLE_LCCL").empty()) {
    std::string lowlatency_comm_lib_name = GetCurrentDir() + "/ascend/liblowlatency_collective.so";
    auto loader = std::make_shared<CollectiveCommLibLoader>(lowlatency_comm_lib_name);
    MS_EXCEPTION_IF_NULL(loader);
    if (!loader->Initialize()) {
      MS_LOG(EXCEPTION) << "Loading LCCL collective library failed.";
      return false;
    }
    void *collective_comm_lib_handle = loader->collective_comm_lib_ptr();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_handle);

    auto instance_func = DlsymFuncObj(communication_lib_instance, collective_comm_lib_handle);
    collective_comm_lib_ = instance_func();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_);
    MS_LOG(WARNING) << "Loading LCCL because env MS_ENABLE_LCCL is set to 1. Pay attention that LCCL only supports "
                       "single-node-multi-card mode in KernelByKernel for now.";
  } else {
    collective_comm_lib_ = &AscendCollectiveCommLib::GetInstance();
  }
  return true;
}

bool GeDeviceResManager::BindDeviceToCurrentThread(bool force_bind) const {
  static thread_local std::once_flag is_set;
  std::call_once(is_set, []() {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(device_id));
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtSetDevice failed, ret:" << static_cast<int>(ret);
    }
    transform::AclUtil::SetDeterministic();
  });

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

size_t GeDeviceResManager::QueryStreamSize() const { return AscendStreamMng::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> GeDeviceResManager::GetStreamIds() const { return AscendStreamMng::GetInstance().GetStreamIds(); }

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

// ACL_EVENT_TIME_LINE: indicates that the number of created events is not limited, and the created events can be used
//  to compute the elapsed time between events, which may cause lost some performance.
// ACL_EVENT_SYNC: indicates that the number of created events is limited, and the created events can be used for
//  synchronization between multiple streams.
// ACL_EVENT_CAPTURE_STREAM_PROGRESS: indicates that the number of created events is not limited and high performance,
//  and the created events can not be used for timing and synchronization.
DeviceEventPtr GeDeviceResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  if (!enable_blocking && !enable_record_wait) {
    MS_LOG(INTERNAL_EXCEPTION) << "Bad parameters, enable_blocking is false and enable_record_wait is false.";
  }

  uint32_t flag = 0;
  if (enable_blocking) {
    flag |= ACL_EVENT_SYNC;
  }
  if (enable_record_wait) {
    flag |= ACL_EVENT_CAPTURE_STREAM_PROGRESS;
  }
  return std::make_shared<AscendEvent>(flag);
}

DeviceEventPtr GeDeviceResManager::CreateEventWithFlag(bool enable_timing, bool blocking) {
  auto flag = enable_timing ? ACL_EVENT_TIME_LINE : ACL_EVENT_DEFAULT;
  auto event = std::make_shared<AscendEvent>(flag);
  MS_EXCEPTION_IF_NULL(event);
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  device_events_.push_back(event);
  return event;
}

void GeDeviceResManager::MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor,
                                const std::string &to, bool blocking, bool *return_self) {
  device::MoveTo(src_tensor, dst_tensor, to, blocking, return_self);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
