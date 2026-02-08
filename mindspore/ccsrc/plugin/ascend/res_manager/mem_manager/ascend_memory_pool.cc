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

#include "plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <set>

#include "tools/profiler/profiling_data_dumper.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/mstx/mstx_impl.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/utils/comm_manager.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "utils/distributed_meta.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr size_t kByteOffset = 8;

struct AscendMemoryTimeEvent : profiler::ascend::BaseReportData {
  explicit AscendMemoryTimeEvent(int32_t device_id, const MemoryTimeEventPtr &memory_time_event);
  virtual ~AscendMemoryTimeEvent() = default;

  std::vector<uint8_t> encode() override;

  uint64_t tid_{0};

  uint64_t pid_{0};

  void *stream_ptr_{nullptr};

  MemoryTimeEventPtr memory_time_event_{nullptr};

  std::string ToJson() {
    JsonBuilder builder;
    builder.Append("tid_", tid_);
    builder.Append("pid_", pid_);
    builder.Append("stream_ptr_", stream_ptr_);
    builder.Append("memory_time_event_", memory_time_event_ ? memory_time_event_->ToJson() : nullptr);
    return builder.ToString();
  }
};
using AscendMemoryTimeEventPtr = std::shared_ptr<AscendMemoryTimeEvent>;

DefaultAscendMemoryPool::DefaultAscendMemoryPool() {
  MS_LOG(DEBUG) << "DefaultAscendMemoryPool constructed.";
  SetEnableVmm(AscendVmmAdapter::IsEnabled());
}

size_t DefaultAscendMemoryPool::EmptyCache() {
  LockGuard lock(AbstractDynamicMemPool::lock());
  AbstractEnhancedDynamicMemPool::WaitPipelineHelper();
  AbstractAscendMemoryPoolSupport::SyncAllStreams();
  size_t release_free_size = 0;
  if (MS_UNLIKELY(!customized_allocators_.empty())) {
    release_free_size += ReleaseCustomFreeBlocks();
  }
  if (IsEnableVmm()) {
    AbstractEnhancedDynamicMemPool::FreeIdleMemsByEagerFree();
    release_free_size += AbstractAscendMemoryPoolSupport::EmptyCache();
    return release_free_size;
  } else if (IsEnableEagerFree()) {
    auto ret = AbstractEnhancedDynamicMemPool::FreeIdleMemsByEagerFree();
    MS_LOG(INFO) << "Eager free memory size is " << ret.second << ".";
    release_free_size += ret.second;
    return release_free_size;
  }

  MS_LOG(INFO) << "Vmm is not enabled, try to release free blocks.";
  // disable ge kernel use two pointer mem adapter, not support free.
  if (IsDisableGeKernel()) {
    return 0L;
  }
  release_free_size += ReleaseFreeBlocks();
  return release_free_size;
}

int32_t GetDeviceId() {
  static const int32_t device_id = []() {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    return ms_context->get_param<int>(MS_CTX_DEVICE_ID);
  }();

  return device_id;
}

MemBufAllocatorPtr DefaultAscendMemoryPool::GenerateCustomAllocator(uint32_t stream_id) {
  MS_LOG(INFO) << "GenerateCustomAllocator, stream id : " << stream_id << ".";
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  std::function<MemBlock *(size_t)> mem_block_expander = [&, stream = stream](size_t size) {
    MemBlock *mem_block = nullptr;
    DeviceMemPtr addr = nullptr;
    MS_LOG(INFO) << "DefaultAscendMemoryPool::Malloc mem block, is enable eager free : " << IsEnableEagerFree()
                 << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ".";

    auto device_id = GetDeviceId();
    addr = custom_alloc_fn_(size, device_id, stream);
    if (addr == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to alloc memory from custom allocator, the addr is nullptr! Please check the alloc "
                           "function in the so which passed to PluggableAllocator";
    }

    mem_stat_ptr_->custom_alloc_size_ += size;
    mem_block = new MemBlock(size, addr, stream_id);
    return mem_block;
  };

  std::function<bool(MemBlock *)> mem_block_cleaner = [&, stream = stream](MemBlock *mem_block) {
    mem_stat_ptr_->custom_alloc_size_ -= mem_block->size_;
    auto device_id = GetDeviceId();
    custom_free_fn_(mem_block->addr_, mem_block->size_, device_id, stream);
    return true;
  };
  std::function<size_t(size_t size, void *addr)> mem_mapper = [](size_t size, void *addr) { return size; };
  std::function<size_t(void *addr, const size_t size)> mem_eager_freer = [](void *addr, const size_t size) {
    return size;
  };

  return std::make_shared<MemBufAllocator>(mem_block_expander, mem_block_cleaner, mem_mapper, mem_eager_freer,
                                           IsEnableEagerFree() || IsEnableVmm(), false, stream_id, false, mem_stat_ptr_,
                                           true);
}

void DefaultAscendMemoryPool::EnablePluggableAllocator(std::function<MallocFuncType> alloc_fn,
                                                       std::function<FreeFuncType> free_fn) {
  custom_alloc_fn_ = alloc_fn;
  custom_free_fn_ = free_fn;
  enable_custom_allocator_ = true;
}

void DefaultAscendMemoryPool::DisablePluggableAllocator() { enable_custom_allocator_ = false; }

AscendMemoryTimeEvent::AscendMemoryTimeEvent(int32_t device_id, const MemoryTimeEventPtr &memory_time_event)
    : BaseReportData(device_id, static_cast<uint32_t>(profiler::ascend::ReportFileType::MEMORY_USAGE)),
      memory_time_event_(memory_time_event) {
  stream_ptr_ = AscendStreamMng::GetInstance().GetStream(memory_time_event_->stream_id_);
}

namespace {
template <typename T>
void EncodeIntoUInt8(T data, std::vector<uint8_t> *result) {
  for (size_t i = 0; i < sizeof(T); i++) {
    result->push_back((static_cast<size_t>(data) >> (i * kByteOffset)) & 0xff);
  }
}

void EncodeStringIntoUInt8(std::string str, std::vector<uint8_t> *result) {
  uint16_t str_type = static_cast<uint16_t>(profiler::ascend::OpRangeDataType::NAME);
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    result->push_back((str_type >> (i * kByteOffset)) & 0xff);
  }
  uint32_t length = str.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    result->push_back((length >> (i * kByteOffset)) & 0xff);
  }
  result->insert(result->end(), str.begin(), str.end());
}

static uint64_t GetTid() {
#ifdef __GNUC__
  static thread_local uint64_t tid = static_cast<uint64_t>(syscall(SYS_gettid));
#else
  static thread_local uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
  return tid;
}

static uint64_t GetPid() {
  static thread_local uint64_t pid = static_cast<uint64_t>(getpid());
  return pid;
}

void FillTidAndPid(const std::unique_ptr<AscendMemoryTimeEvent> &ascend_mmemory_time_event) {
  ascend_mmemory_time_event->tid_ = GetTid();
  ascend_mmemory_time_event->pid_ = GetPid();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Fill time event info : " << ascend_mmemory_time_event->ToJson() << ".";
}
}  // namespace

std::vector<uint8_t> AscendMemoryTimeEvent::encode() {
  std::vector<uint8_t> result;
  EncodeIntoUInt8<int32_t>(device_id, &result);
  EncodeIntoUInt8<uint64_t>(tid_, &result);
  EncodeIntoUInt8<uint64_t>(pid_, &result);
  EncodeIntoUInt8<uint64_t>(memory_time_event_->created_at_, &result);
  EncodeIntoUInt8<size_t>(reinterpret_cast<size_t>(memory_time_event_->addr_), &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->used_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->peak_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->alloc_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->used_by_event_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->eager_free_size_, &result);
  EncodeIntoUInt8<size_t>(reinterpret_cast<size_t>(stream_ptr_), &result);
  EncodeIntoUInt8<uint32_t>(memory_time_event_->stream_id_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->from_persistent_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->is_persistent_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->run_mode_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->alloc_type_, &result);
  EncodeStringIntoUInt8(memory_time_event_->owner_, &result);

  std::vector<uint8_t> tlv_result;
  uint16_t data_type = static_cast<uint16_t>(profiler::ascend::OpRangeDataType::NAME);
  for (size_t i = 0; i < sizeof(uint16_t); i++) {
    (void)tlv_result.emplace_back((data_type >> (i * kByteOffset)) & 0xff);
  }
  uint32_t length = result.size();
  for (size_t i = 0; i < sizeof(uint32_t); i++) {
    (void)tlv_result.emplace_back((length >> (i * kByteOffset)) & 0xff);
  }
  tlv_result.insert(tlv_result.end(), result.cbegin(), result.cend());
  return tlv_result;
}

DefaultEnhancedAscendMemoryPool::DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr &instance)
    : instance_(instance) {
  MS_LOG(INFO) << "DefaultEnhancedAscendMemoryPool constructed.";
  instance_->SetEnableVmm(AscendVmmAdapter::IsEnabled());
}

void DefaultEnhancedAscendMemoryPool::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Start release device res.";
  instance_->ReleaseDeviceRes();
}

DeviceMemPtr DefaultEnhancedAscendMemoryPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool need_recycle,
                                                             uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Allocate tensor mem, size : " << size << ", align_size : " << align_size
                                       << ", need_recycle : " << need_recycle << ".";
  LockGuard lock(instance_->lock());
  const auto [mem_buf, allocator] = instance_->AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (mem_buf == nullptr) {
    MS_LOG(INFO) << "Allocate tensor mem, return nullptr.";
    // Dump mem pool state info and debug info when alloc tensor failed.
    DumpDynamicMemPoolStateInfo();
    DumpDynamicMemPoolDebugInfo();
    return nullptr;
  }

  mem_buf->SetDebugInfo();
  instance_->addr_mem_buf_allocators().emplace(mem_buf->addr_, std::make_pair(mem_buf, allocator));
  auto device_addr = mem_buf->addr_;

  instance_->ReportMemoryPoolInfo();
  instance_->ReportMemoryPoolMallocInfoToMstx(device_addr, align_size);

  // Adapt for dry run.
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, Memory pool alloc, total mem: " << TotalMemStatistics()
                    << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                    << ", used by event mem: " << TotalUsedByEventMemStatistics()
                    << ", device address addr: " << device_addr << ", size: " << align_size
                    << ", from persistent mem: " << from_persistent_mem << ", need recycle: " << need_recycle << ".";
  }

  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER(AllocMemBlock, device_addr, mem_buf->size_, GetMemoryPoolType(),
                                 ActualPeakStatistics(), TotalUsedMemStatistics(), stream_id, from_persistent_mem,
                                 UseSmallPool(mem_buf->size_, from_persistent_mem));
  }

  // Time line process.
  if (instance_->IsEnableTimeEvent()) {
    int32_t device_id = GetDeviceId();
    auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(
      device_id, instance_->GenAllocateMemoryTimeEvent(mem_buf->addr_, mem_buf->size_, stream_id, from_persistent_mem,
                                                       allocator->is_persistent()));
    ascend_memory_time_event->stream_ptr_ =
      AscendStreamMng::GetInstance().GetStream(ascend_memory_time_event->memory_time_event_->stream_id_);
    FillTidAndPid(ascend_memory_time_event);
    profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
  }

  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Allocate tensor mem, return : " << mem_buf->ToJson()
                                       << ", stat info : " << instance_->mem_stat_ptr()->ToJson() << ".";
  return device_addr;
}

std::vector<DeviceMemPtr> DefaultEnhancedAscendMemoryPool::AllocContinuousTensorMem(
  const std::vector<size_t> &size_list, uint32_t stream_id) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Alloc continuous tensor mem, stream id : " << stream_id << ".";
  const auto &continuous_addrs = instance_->AllocContinuousTensorMem(size_list, stream_id);
  if (continuous_addrs.size() != size_list.size()) {
    return continuous_addrs;
  }
  if (continuous_addrs.size() == 1 && continuous_addrs[0] == nullptr) {
    return continuous_addrs;
  }

  for (size_t i = 0; i < continuous_addrs.size(); i++) {
    if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      // AllocContinuousTensorMem allocates a large continuous MemBuf first and then split them into small MemBufs
      // so the total size of MemBuf determines small/large allocator it uses.
      const size_t total_size = std::accumulate(size_list.begin(), size_list.end(), static_cast<size_t>(0));
      tracker::CALL_MEMORY_TRACKER(AllocMemBlock, continuous_addrs[i], size_list[i], GetMemoryPoolType(),
                                   ActualPeakStatistics(), TotalUsedMemStatistics(), stream_id, false,
                                   UseSmallPool(total_size, false));
    }

    if (instance_->IsEnableTimeEvent()) {
      int32_t device_id = GetDeviceId();
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(
        device_id, instance_->GenAllocateMemoryTimeEvent(continuous_addrs[i], size_list[i], stream_id, false, false));
      ascend_memory_time_event->stream_ptr_ =
        AscendStreamMng::GetInstance().GetStream(ascend_memory_time_event->memory_time_event_->stream_id_);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }
  return continuous_addrs;
}

void DefaultEnhancedAscendMemoryPool::FreeTensorMem(const DeviceMemPtr &device_addr) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Free tensor mem, device addr : " << device_addr << ".";
  LockGuard lock(instance_->lock());
  DoFreeTensorMem(device_addr);
}

bool DefaultEnhancedAscendMemoryPool::IsNotEventUsedTensorMem(const DeviceMemPtr &device_addr) {
  LockGuard lock(instance_->lock());
  return instance_->IsNotEventUsedTensorMem(device_addr);
}

bool DefaultEnhancedAscendMemoryPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *enhanced_device_addr = device_addr;
  bool ret = instance_->DoFreeTensorMem(device_addr);
  if (ret) {
    instance_->ReportMemoryPoolInfo();
    instance_->ReportMemoryPoolFreeInfoToMstx(enhanced_device_addr);

    // Adapt for dry run.
    if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
      MS_LOG(WARNING) << "Need Profile Memory, Memory pool free, total mem: " << TotalMemStatistics()
                      << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                      << ", used by event mem: " << TotalUsedByEventMemStatistics()
                      << ", device address addr: " << enhanced_device_addr << ".";
    }

    // Adapt for mem tracker.
    if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      tracker::CALL_MEMORY_TRACKER(FreeMemBlock, enhanced_device_addr, TotalUsedMemStatistics(), TotalMemStatistics());
    }

    if (instance_->IsEnableTimeEvent()) {
      int32_t device_id = GetDeviceId();
      auto time_event = GenFreeMemoryTimeEvent(enhanced_device_addr);
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(device_id, time_event);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Do free tensor mem : " << enhanced_device_addr << ", return : " << ret
                                       << ".";
  return ret;
}

void DefaultEnhancedAscendMemoryPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                         const std::vector<DeviceMemPtr> &keep_addrs,
                                                         const std::vector<size_t> &keep_addr_sizes) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Free part tensor mems.";
  LockGuard lock(instance_->lock());
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    for (const auto &free_addr : free_addrs) {
      tracker::CALL_MEMORY_TRACKER(FreeMemBlock, free_addr, TotalUsedMemStatistics(), TotalMemStatistics());
    }
  }

  if (instance_->IsEnableTimeEvent()) {
    int32_t device_id = GetDeviceId();
    for (const auto &free_addr : free_addrs) {
      auto time_event = GenFreeMemoryTimeEvent(free_addr);
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(device_id, time_event);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }

  const auto keep_mem_bufs = instance_->DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    for (const auto &mem_buf : keep_mem_bufs) {
      if (const auto &&it = addr_mem_buf_allocators_.find(mem_buf->addr_); it != addr_mem_buf_allocators_.end()) {
        const auto mem_buf_allocator = it->second.second;
        MS_EXCEPTION_IF_NULL(mem_buf_allocator);
        // DoFreePartTensorMems splits a large MemBuf, keeps a part of it and free the rest, so whether the allocator
        // is small or persistent depends on the size of the large MemBuf not the one that is kept. Here, we just use
        // the info stored in the allocator.
        const bool is_small = mem_buf_allocator->is_small();
        const bool is_persistent = mem_buf_allocator->is_persistent();
        tracker::CALL_MEMORY_TRACKER(AllocMemBlock, mem_buf->addr_, mem_buf->size_, GetMemoryPoolType(),
                                     ActualPeakStatistics(), TotalUsedMemStatistics(), mem_buf->stream_id_,
                                     is_persistent, is_small);
      } else {
        MS_LOG(DEBUG) << "Find mem buf address: " << mem_buf->addr_ << " failed.";
      }
    }
  }
  if (instance_->IsEnableTimeEvent()) {
    int32_t device_id = GetDeviceId();
    for (const auto &mem_buf : keep_mem_bufs) {
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(
        device_id,
        instance_->GenAllocateMemoryTimeEvent(mem_buf->addr_, mem_buf->size_, mem_buf->stream_id_, false, false));
      ascend_memory_time_event->stream_ptr_ =
        AscendStreamMng::GetInstance().GetStream(ascend_memory_time_event->memory_time_event_->stream_id_);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }
}

void DefaultEnhancedAscendMemoryPool::DefragMemory() {
  if (last_vmm_used_size_ == 0) {
    last_vmm_used_size_ = GetVmmUsedMemSize();
  } else {
    size_t vmm_used_size = GetVmmUsedMemSize();
    if (vmm_used_size > last_vmm_used_size_) {
      MS_LOG(WARNING) << "Current vmm used size : " << vmm_used_size
                      << " is bigger than last vmm used size : " << last_vmm_used_size_ << ".";
      last_vmm_used_size_ = vmm_used_size;
    }
  }

  instance_->DefragMemory();
}

void DefaultEnhancedAscendMemoryPool::DumpDynamicMemPoolStateInfo() {
  const auto &state_info = instance_->DynamicMemPoolStateInfo();
  static bool is_enable_memory_statistics = runtime::IsEnableRuntimeConfig(runtime::kRuntimeMemoryStat);
  if (is_enable_memory_statistics) {
    std::cout << "[MS_RUNTIME_PROF]" << state_info << std::endl;
  }
  instance_->DumpDynamicMemPoolStateInfo();
}

const std::pair<size_t, size_t> DefaultEnhancedAscendMemoryPool::FreeIdleMemsByEagerFree() {
  const auto [eager_free_size, real_free_size] = instance_->FreeIdleMemsByEagerFree();
  static bool is_enable_memory_statistics = runtime::IsEnableRuntimeConfig(runtime::kRuntimeMemoryStat);
  if (is_enable_memory_statistics) {
    std::cout << "Total eager free memory : " << eager_free_size << ", real free : " << real_free_size << "."
              << std::endl;
  }
  return {eager_free_size, real_free_size};
}

// Element in vector : <memory_stream_id, addr>
bool DefaultEnhancedAscendMemoryPool::RecordEvent(
  int64_t task_id_on_stream, uint32_t user_stream_id,
  const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses, const DeviceEventPtr &event) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Record event for task id on stream : " << task_id_on_stream
                                       << ", user stream id : " << user_stream_id << " event : " << event << ".";
  LockGuard lock(instance_->lock());
  for (auto &[memory_stream_id, addr] : memory_stream_addresses) {
    auto &&it = instance_->addr_mem_buf_allocators().find(addr);
    if (it != instance_->addr_mem_buf_allocators().end()) {
      auto mem_buf = it->second.first;
      if (mem_buf->IsEventNotUsed()) {
        instance_->mem_stat_ptr()->used_by_event_size_ += mem_buf->size_;
      }
      (void)mem_buf->RecordEvent(task_id_on_stream, user_stream_id, event);
      (void)instance_->stream_pair_mem_bufs()[std::make_pair(user_stream_id, memory_stream_id)].emplace(mem_buf);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY)
        << "Record event for : " << mem_buf->ToJson() << ", memory_stream_id : " << memory_stream_id
        << ", events : " << mem_buf->Events() << ".";
    } else {
      // Output of somas sub graph may be used by somas sub graph inner node, address may not be kept in mem pool.
      MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Unknown address : " << addr << ".";
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                                uint32_t memory_stream_id) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Wait event for task id on stream : " << task_id_on_stream
                                       << ", user stream id : " << user_stream_id
                                       << ", memory stream id : " << memory_stream_id << ".";
  LockGuard lock(instance_->lock());
  auto key = std::make_pair(user_stream_id, memory_stream_id);
  auto iter = instance_->stream_pair_mem_bufs().find(key);
  if (iter == instance_->stream_pair_mem_bufs().end()) {
    return false;
  }

  auto mem_bufs_ = iter->second;
  for (const auto &mem_buf : mem_bufs_) {
    mem_buf->WaitEvent(task_id_on_stream, user_stream_id);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Wait event for : " << mem_buf->ToJson()
                                         << ", events : " << mem_buf->Events() << ".";
    // Remove event and try to free memory.
    if (mem_buf->IsEventNotUsed()) {
      instance_->mem_stat_ptr()->used_by_event_size_ -= mem_buf->size_;
      // Force clear all mem bufs.
      for (auto &stream_pair_mem_bufs : instance_->stream_pair_mem_bufs()) {
        (void)stream_pair_mem_bufs.second.erase(mem_buf);
      }
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        (void)DoFreeTensorMem(mem_buf->addr_);
      }
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Wait event for task id on stream : " << task_id_on_stream
                                       << ", memory stream id : " << memory_stream_id << ".";
  LockGuard lock(instance_->lock());
  for (auto &stream_pair_mem_bufs : instance_->stream_pair_mem_bufs()) {
    const auto &[user_stream, memory_stream] = stream_pair_mem_bufs.first;
    if (memory_stream != memory_stream_id) {
      continue;
    }
    auto mem_bufs = stream_pair_mem_bufs.second;
    for (const auto &mem_buf : mem_bufs) {
      mem_buf->WaitEvent(task_id_on_stream, user_stream);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY)
        << "Wait event for : " << mem_buf->ToJson() << ", events : " << mem_buf->Events() << ".";
      // Remove event and try to free memory.
      if (mem_buf->IsEventNotUsed()) {
        instance_->mem_stat_ptr()->used_by_event_size_ -= mem_buf->size_;
        // Force clear all mem bufs.
        for (auto &kv : instance_->stream_pair_mem_bufs()) {
          (void)kv.second.erase(mem_buf);
        }
        if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
          (void)DoFreeTensorMem(mem_buf->addr_);
        }
      }
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::SyncAllEvents() {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Sync all events, stream_pair_addresses_ size : "
                                       << instance_->stream_pair_mem_bufs().size() << ".";
  return instance_->SyncAllEvents();
}

void DefaultEnhancedAscendMemoryPool::SetRankIdGetter(const std::function<size_t()> &rank_id_getter) {
  instance_->SetRankIdGetter(rank_id_getter);
  if (rank_id_getter != nullptr) {
    rank_id_getter_ = rank_id_getter;
  }
}

BestFitAscendMemoryPool::BestFitAscendMemoryPool() {
  MS_LOG(INFO) << "BestFitAscendMemoryPool constructed, older memory allocator is enabled.";
  SetEnableVmm(AscendVmmAdapter::IsEnabled());
}

void BestFitAscendMemoryPool::ReportMemoryTimeEvent(const MemoryTimeEventPtr &time_event) {
  int32_t device_id = GetDeviceId();
  auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(device_id, time_event);
  if (time_event->stream_id_ != UINT32_MAX) {
    ascend_memory_time_event->stream_ptr_ = AscendStreamMng::GetInstance().GetStream(time_event->stream_id_);
  }
  FillTidAndPid(ascend_memory_time_event);
  profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
}

size_t BestFitAscendMemoryPool::EmptyCache() {
  MS_LOG(WARNING) << "Best fit memory pool is not supported empty cache.";
  return 0L;
}

// Initialize static member in AscendMemoryPool.
AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::pool_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::instance_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::enhanced_instance_ = nullptr;

AbstractAscendMemoryPoolSupport &AscendMemoryPool::GetInstance() {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    if (UseOldMemoryPool()) {
      instance_ = std::make_shared<BestFitAscendMemoryPool>();
      enhanced_instance_ = instance_;
    } else {
      const auto &memory_pool = std::make_shared<DefaultAscendMemoryPool>();
      instance_ = memory_pool;
      enhanced_instance_ = std::make_shared<DefaultEnhancedAscendMemoryPool>(memory_pool);
    }
    // Initialize instance and set ptr.
    float init_size = runtime::RuntimeConf::GetInstance()->mem_init_size();
    size_t init_size_byte = FloatToSize(init_size * kGBToByte);
    float increase_size = runtime::RuntimeConf::GetInstance()->mem_block_increase_size();
    size_t increase_size_byte = FloatToSize(increase_size * kGBToByte);
    float max_size = runtime::RuntimeConf::GetInstance()->mem_max_size();
    size_t max_size_byte = FloatToSize(max_size * kGBToByte);
    instance_->Initialize(init_size_byte, increase_size_byte, max_size_byte);
#ifdef ENABLE_DEBUGGER
    // Set memory profiler callback func.
    instance_->SetMemoryProfilerCallback([&]() {
      static auto profiler_inst = profiler::Profiler::GetInstance(kCPUDevice);
      MS_EXCEPTION_IF_NULL(profiler_inst);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Start report memory pool info.";
      if (profiler_inst->GetEnableFlag() && profiler_inst->GetProfileMemoryFlag()) {
        profiler_inst->RecordMemoryPoolInfo(instance_->TotalUsedMemStatistics(), instance_->TotalMemStatistics(),
                                            instance_->TotalUsedByEventMemStatistics());
      }
    });
#endif
    // Set memory mstx callback func.
    instance_->SetMemoryMstxCallback(
      [&](void *addr, size_t size) {
        if (profiler::MstxImpl::GetInstance().IsMsleaksEnable()) {
          uint32_t device_id = GetDeviceId();
          profiler::mstxDomainHandle_t msleaksDomain =
            profiler::MstxImpl::GetInstance().DomainCreateAImpl(profiler::MSTX_DOMAIN_MSLEAKS);
          profiler::mstxMemVirtualRangeDesc_t desc{device_id, addr, static_cast<int64_t>(size)};
          profiler::mstxMemRegionsRegisterBatch_t batch;
          batch.regionCount = 1;
          batch.regionDescArray = reinterpret_cast<const void *>(&desc);
          profiler::MstxImpl::GetInstance().MemRegionsRegisterImpl(msleaksDomain, &batch);

          profiler::mstxMemVirtualRangeDesc_t descTotal{device_id, addr,
                                                        static_cast<int64_t>(instance_->TotalMemStatistics())};
          profiler::mstxMemHeapDesc_t heapDesc;
          heapDesc.typeSpecificDesc = reinterpret_cast<void const *>(&descTotal);
          profiler::MstxImpl::GetInstance().MemHeapRegisterImpl(msleaksDomain, &heapDesc);
        }
      },
      [&](void *addr) {
        if (profiler::MstxImpl::GetInstance().IsMsleaksEnable()) {
          uint32_t device_id = GetDeviceId();
          profiler::mstxDomainHandle_t msleaksDomain =
            profiler::MstxImpl::GetInstance().DomainCreateAImpl(profiler::MSTX_DOMAIN_MSLEAKS);
          profiler::mstxMemRegionsUnregisterBatch_t unregisterBatch;
          unregisterBatch.refCount = 1;
          profiler::mstxMemRegionRef_t regionRef[1] = {};
          regionRef[0].refType = profiler::MSTX_MEM_REGION_REF_TYPE_POINTER;
          regionRef[0].pointer = addr;
          unregisterBatch.refArray = regionRef;
          profiler::MstxImpl::GetInstance().MemRegionsUnregisterImpl(msleaksDomain, &unregisterBatch);

          profiler::mstxMemVirtualRangeDesc_t descTotal{device_id, addr,
                                                        static_cast<int64_t>(instance_->TotalMemStatistics())};
          profiler::mstxMemHeapDesc_t heapDesc;
          heapDesc.typeSpecificDesc = reinterpret_cast<void const *>(&descTotal);
          profiler::MstxImpl::GetInstance().MemHeapRegisterImpl(msleaksDomain, &heapDesc);
        }
      });

    enhanced_instance_->SetRankIdGetter([]() {
      size_t rank_id = SIZE_MAX;
      if (DistributedMeta::GetInstance()->initialized()) {
        rank_id = DistributedMeta::GetInstance()->global_rank_id();
      }
      return rank_id;
    });
    instance_->SetPipelineCallback([]() { runtime::Pipeline::Get().launch_stage()->Wait(); });
    if (!UseEnhancedMemoryPool()) {
      pool_ = instance_;
      device::tracker::CALL_MEMORY_TRACKER(SetEnableMemoryDebugInfo, false);
    } else {
      pool_ = enhanced_instance_;
      device::tracker::CALL_MEMORY_TRACKER(SetEnableMemoryDebugInfo, true);
    }
  });
  return *pool_;
}

void AscendMemoryPool::SetEnhancedMemoryPool(bool enable) {
  MS_LOG(INFO) << "Set enhanced memory pool : " << enable << ".";
  if (enable) {
    pool_ = enhanced_instance_;
  } else {
    pool_ = instance_;
  }
  device::tracker::CALL_MEMORY_TRACKER(SetEnableMemoryDebugInfo, enable);
}

bool AscendMemoryPool::UseOldMemoryPool() {
  if (memory::mem_pool::IsDisableAllocConfig(memory::mem_pool::kAllocMemoryPool)) {
    return false;
  }
  return IsDisableGeKernel() || memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocMemoryPool);
}

// Use enhanced memory pool when enable debug, enable log, enable prof, dry run and so on.
bool AscendMemoryPool::UseEnhancedMemoryPool() {
  bool enable_debugger = false;
#ifdef ENABLE_DEBUGGER
  auto profiler = profiler::Profiler::GetInstance(kCPUDevice);
  if (profiler != nullptr && profiler->GetEnableFlag() && profiler->GetProfileMemoryFlag()) {
    enable_debugger = true;
  }
#endif
  bool enable_debug_log = common::GetEnv("GLOG_v") == "0";
  bool enable_memory_vlog = IS_VLOG_ON(VL_RUNTIME_FRAMEWORK_MEMORY);
  return enable_debugger || enable_debug_log || enable_memory_vlog || common::IsCompileSimulation() ||
         profiler::MstxImpl::GetInstance().IsMsleaksEnable() || memory::mem_pool::IsEnableMemTrack();
}

std::string AscendMemoryPool::ParseDebugConfig(std::string input, std::string config) {
  auto pos = input.find(config);
  if (pos == std::string::npos) {
    return "";
  }
  auto config_pos = input.find(",", pos);
  size_t skip_count = config.size() + 1;
  auto config_str = input.substr(pos + skip_count, config_pos - pos - skip_count);
  if (config_str.find("}") != std::string::npos) {
    config_str = config_str.substr(0, config_str.size() - 1);
  }
  // need trim laster
  return config_str;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
