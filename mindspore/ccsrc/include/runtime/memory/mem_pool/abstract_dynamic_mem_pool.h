/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_ABSTRACT_DYNAMIC_MEM_POOL_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_ABSTRACT_DYNAMIC_MEM_POOL_H_

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/runtime/memory/mem_pool/dynamic_mem_pool.h"
#include "include/backend/visible.h"
#include "include/utils/stream_util.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
constexpr size_t kDecimalPrecision = 3;
// largest allocation size for small pool is 1 MB
constexpr size_t kSmallSize = 1048576;

/// @brief Check if small pool environment variable is enabled.
///
/// @return True if small pool is enabled, false otherwise.
inline bool IsEnableSmallPool() {
  static const bool is_enable_small_pool = [] {
    return memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocEnableSmallPool);
  }();
  return is_enable_small_pool;
}

constexpr size_t kPoolGrowSize = 1 << 20;

template <class T>
class ObjectPool {
  struct Buf {
    Buf *next_;
  };

  class Buffer {
    static const std::size_t bucket_size = sizeof(T) > sizeof(Buf) ? sizeof(T) : sizeof(Buf);
    static const std::size_t kDataBucketSize = bucket_size * kPoolGrowSize;

   public:
    explicit Buffer(Buffer *next) : next_(next) {}

    T *GetBlock(std::size_t index) {
      if (index >= kPoolGrowSize) {
        throw std::bad_alloc();
      }
      return reinterpret_cast<T *>(&data_[bucket_size * index]);
    }

    Buffer *const next_;

   private:
    uint8_t data_[kDataBucketSize];
  };

  Buf *free_list_ = nullptr;
  Buffer *buffer_head_ = nullptr;
  std::size_t buffer_index_ = kPoolGrowSize;

 public:
  ObjectPool() = default;
  ObjectPool(ObjectPool &&object_pool) = delete;
  ObjectPool(const ObjectPool &object_pool) = delete;
  ObjectPool operator=(const ObjectPool &object_pool) = delete;
  ObjectPool operator=(ObjectPool &&object_pool) = delete;

  ~ObjectPool() {
    while (buffer_head_ != nullptr) {
      Buffer *buffer = buffer_head_;
      buffer_head_ = buffer->next_;
      delete buffer;
    }
  }

  T *Borrow() {
    if (free_list_ != nullptr) {
      Buf *buf = free_list_;
      free_list_ = buf->next_;
      return reinterpret_cast<T *>(buf);
    }

    if (buffer_index_ >= kPoolGrowSize) {
      buffer_head_ = new Buffer(buffer_head_);
      buffer_index_ = 0;
    }

    return buffer_head_->GetBlock(buffer_index_++);
  }

  void Return(T *obj) {
    Buf *buf = reinterpret_cast<Buf *>(obj);
    buf->next_ = free_list_;
    free_list_ = buf;
  }
};

// Not support older windows version.
template <class T>
class PooledAllocator : private ObjectPool<T> {
 public:
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef T value_type;

  template <class U>
  struct rebind {
    typedef PooledAllocator<U> other;
  };

  pointer allocate(size_type n, const void *hint = 0) {
    if (n != 1 || hint) throw std::bad_alloc();
    return ObjectPool<T>::Borrow();
  }

  void deallocate(pointer p, size_type n) { ObjectPool<T>::Return(p); }

  void construct(pointer p, const_reference val) { new (p) T(val); }

  void destroy(pointer p) { p->~T(); }
};

struct BACKEND_EXPORT MemBlock;

using MemBufStatus = DynamicMemBufStatus;
struct BACKEND_EXPORT MemBuf : EventBase {
  explicit MemBuf(size_t size, void *addr, uint32_t stream_id, MemBlock *mem_block, MemBufStatus status);

  MemBuf() = delete;
  MemBuf(const MemBuf &) = delete;
  MemBuf &operator=(const MemBuf &) = delete;

  ~MemBuf();

  inline void Link(MemBuf *prev, MemBuf *next) {
    if (prev != nullptr) {
      prev->next_ = this;
      this->prev_ = prev;
    }
    if (next != nullptr) {
      next->prev_ = this;
      this->next_ = next;
    }
  }

  inline void Unlink() {
    if (prev_ != nullptr) {
      prev_->next_ = next_;
    }
    if (next_ != nullptr) {
      next_->prev_ = prev_;
    }
    prev_ = nullptr;
    next_ = nullptr;
  }

  inline void SetDebugInfo() {
    owner_name_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
    alloc_type_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().type_;
  }

  std::string ToJson() {
    JsonBuilder builder;
    builder.Append("addr_", addr_);
    builder.Append("size_", size_);
    builder.Append("stream_id_", stream_id_);
    builder.Append("status_", DynamicMemBufStatusToString(status_));
    builder.Append("owner_name_", owner_name_);
    builder.Append("alloc_type_", MemTypeToStr(alloc_type_));
    return builder.ToString();
  }

  std::string Events() {
    std::stringstream ss;
    for (const auto &event : *events_) {
      ss << "usr_stream_id: " << event.first << " {";
      for (const auto &task_id_on_stream : *event.second) {
        ss << "task_id: " << task_id_on_stream.first << ", event: " << task_id_on_stream.second << ", ";
      }
      ss << "}, ";
    }
    return ss.str();
  }

  MemBuf *prev_;
  MemBuf *next_;

  size_t size_;
  void *addr_;
  uint32_t stream_id_;
  MemBlock *mem_block_;
  MemBufStatus status_;
  memory::mem_pool::MemType alloc_type_{memory::mem_pool::MemType::kOther};
  std::string owner_name_;
};

struct MemBufComparator {
  bool operator()(MemBuf *const &left, MemBuf *const &right) const {
    return (left->size_ != right->size_) ? left->size_ < right->size_ : left->addr_ < right->addr_;
  }
};

struct BACKEND_EXPORT MemBlock {
  explicit MemBlock(size_t size, void *addr, uint32_t stream_id) : size_(size), addr_(addr), stream_id_(stream_id) {
    min_addr_ = nullptr;
    max_addr_ = nullptr;
  }

  MemBlock() = delete;
  MemBlock(const MemBlock &) = delete;
  MemBlock &operator=(const MemBlock &) = delete;

  ~MemBlock() = default;

  inline void UpdateBorderAddr(MemBuf *mem_buf) {
    if (min_addr_ == nullptr) {
      min_addr_ = mem_buf->addr_;
    } else {
      min_addr_ = std::min(min_addr_, mem_buf->addr_);
    }
    void *right_addr = static_cast<uint8_t *>(mem_buf->addr_) + mem_buf->size_;
    max_addr_ = std::max(max_addr_, right_addr);
  }

  inline size_t ActualPeakSize() {
    if (min_addr_ == nullptr || max_addr_ == nullptr) {
      return 0;
    }
    return static_cast<uint8_t *>(max_addr_) - static_cast<uint8_t *>(min_addr_);
  }

  std::string ToJson() {
    JsonBuilder builder;
    builder.Append("addr_", addr_);
    builder.Append("size_", size_);
    builder.Append("stream_id_", stream_id_);
    builder.Append("min_addr_", min_addr_);
    builder.Append("max_addr_", max_addr_);
    return builder.ToString();
  }

  size_t size_;
  void *addr_;
  uint32_t stream_id_;

  void *min_addr_;
  void *max_addr_;
};

struct BACKEND_EXPORT MemStat {
  MemStat() { Reset(); }

  MemStat(const MemStat &) = delete;
  MemStat &operator=(const MemStat &) = delete;

  void Reset() {
    used_size_ = 0;
    peak_size_ = 0;
    alloc_size_ = 0;
    custom_alloc_size_ = 0;

    used_by_event_size_ = 0;
    eager_free_size_ = 0;

    iter_used_peak_size_ = 0;
    iter_alloc_peak_size_ = 0;
  }

  size_t IdleSize() const {
    return alloc_size_ + custom_alloc_size_ - used_size_ - eager_free_size_ - used_by_event_size_;
  }

  inline void UpdatePeakSize(const bool is_enable_vmm, size_t vmm_used_mem_size) {
    peak_size_ = std::max(peak_size_, used_size_);
    iter_used_peak_size_ = std::max(iter_used_peak_size_, used_size_);
    if (is_enable_vmm) {
      iter_alloc_peak_size_ = std::max(iter_alloc_peak_size_, vmm_used_mem_size + custom_alloc_size_);
    } else {
      iter_alloc_peak_size_ = std::max(iter_alloc_peak_size_, alloc_size_ + custom_alloc_size_);
    }
  }

  std::string ToJson() const {
    JsonBuilder builder;
    builder.Append("used_size_", used_size_);
    builder.Append("peak_size_", peak_size_);
    builder.Append("alloc_size_", alloc_size_);
    builder.Append("idle_size_", IdleSize());
    builder.Append("used_by_event_size_", used_by_event_size_);
    builder.Append("eager_free_size_", eager_free_size_);
    return builder.ToString();
  }

  std::string ToReadableString() const {
    JsonBuilder builder;
    builder.Append("in used mem", Format(used_size_));
    builder.Append("peak used mem", Format(peak_size_));
    builder.Append("alloc mem", Format(alloc_size_));
    builder.Append("idle mem", Format(IdleSize()));
    builder.Append("used by event mem", Format(used_by_event_size_));
    builder.Append("eager free mem", Format(eager_free_size_));
    return builder.ToString();
  }

  std::string Format(size_t size) const {
    auto str = std::to_string(size * 1.0 / kMBToByte);
    return str.substr(0, str.find(".") + kDecimalPrecision) + "MB";
  }

  size_t used_size_;
  size_t peak_size_;
  size_t alloc_size_;
  size_t custom_alloc_size_;

  size_t used_by_event_size_;
  size_t eager_free_size_;

  size_t iter_used_peak_size_;
  size_t iter_alloc_peak_size_;
};
using MemStatPtr = std::shared_ptr<MemStat>;

struct AllocatorInfo {
  uint32_t stream_id = 0;
  bool from_persistent_mem = false;
  bool use_small_pool = false;

  bool operator<(const AllocatorInfo &other) const {
    if (stream_id != other.stream_id) {
      return stream_id < other.stream_id;
    }
    if (from_persistent_mem != other.from_persistent_mem) {
      return other.from_persistent_mem;
    }
    if (use_small_pool != other.use_small_pool) {
      return other.use_small_pool;
    }
    return false;
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << "stream id: " << stream_id << ", is persistent: " << from_persistent_mem
        << ", use small pool: " << use_small_pool;
    return oss.str();
  }
};

class AbstractDynamicMemPool;

class BACKEND_EXPORT MemBufAllocator {
 public:
  explicit MemBufAllocator(std::function<MemBlock *(size_t)> mem_block_expander,
                           std::function<bool(MemBlock *)> mem_block_cleaner,
                           std::function<size_t(size_t size, void *addr)> mem_mapper,
                           std::function<size_t(void *addr, size_t size)> mem_eager_freer, bool enable_eager_free,
                           bool is_persistent, uint32_t stream_id, bool is_small, MemStatPtr mem_stat_ptr_,
                           bool is_customized = false)
      : mem_block_expander_(std::move(mem_block_expander)),
        mem_block_cleaner_(std::move(mem_block_cleaner)),
        mem_mapper_(std::move(mem_mapper)),
        mem_eager_freer_(std::move(mem_eager_freer)),
        stream_id_(stream_id),
        enable_eager_free_(enable_eager_free),
        is_persistent_(is_persistent),
        is_small_(is_small),
        is_customized_(is_customized),
        mem_stat_ptr_(std::move(mem_stat_ptr_)) {
    search_key_ = new MemBuf(0, nullptr, 0, nullptr, MemBufStatus::kMemBufIdle);
  }

  MemBufAllocator() = delete;
  MemBufAllocator(const MemBufAllocator &) = delete;
  MemBufAllocator &operator=(const MemBufAllocator &) = delete;

  ~MemBufAllocator();

  void Initialize(size_t size);
  void ReleaseDeviceRes();

  MemBuf *Malloc(size_t size);
  MemBuf *SearchAvailableMemBuf(size_t size);
  bool Free(MemBuf *mem_buf, MemBufStatus target_status = MemBufStatus::kMemBufIdle);
  MemBuf *MallocExpandBlock(size_t size);
  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree();

  size_t ReleaseFreeBlocks();

  std::string DumpStateInfo() const;
  std::string DumpDebugInfo() const;

  size_t ActualPeakSize() const {
    size_t peak_size = 0;
    for (auto mem_block : mem_blocks_) {
      peak_size += mem_block->ActualPeakSize();
    }
    return peak_size;
  }

  std::string BriefInfo() const {
    std::stringstream ss;
    ss << "Mem buf allocator, enable vmm : " << enable_eager_free_ << ", is persistent : " << is_persistent_
       << ", stream id : " << stream_id_ << ", is small : " << is_small_ << ", is customized : " << is_customized_
       << ".";
    return ss.str();
  }

  uint32_t stream_id() const { return stream_id_; }
  bool is_persistent() const { return is_persistent_; }
  bool is_small() const { return is_small_; }

#ifndef ENABLE_TEST

 protected:
#endif
  MemBuf *MapAndSplitMemBuf(MemBuf *candidate, size_t size);
  MemBlock *ExpandBlock(size_t size);

  std::function<MemBlock *(size_t)> mem_block_expander_;
  std::function<bool(MemBlock *)> mem_block_cleaner_;
  std::function<size_t(size_t size, void *addr)> mem_mapper_;
  std::function<size_t(void *addr, size_t size)> mem_eager_freer_;

  std::list<MemBlock *> mem_blocks_;
  using MemAllocator = PooledAllocator<MemBuf *>;
  std::set<MemBuf *, MemBufComparator, MemAllocator> free_mem_bufs_;
  std::set<MemBuf *, MemBufComparator, MemAllocator> eager_free_mem_bufs_;

#ifndef ENABLE_TEST

 private:
#endif
  MemBuf *search_key_;

  uint32_t stream_id_;
  bool enable_eager_free_;
  bool is_persistent_;
  bool is_small_;
  bool is_customized_;
  MemStatPtr mem_stat_ptr_;

  void EraseEagerFreeBuf(MemBuf *mem_buf) {
    const auto ret = eager_free_mem_bufs_.erase(mem_buf);
    if (ret == 0) {
      MS_LOG(ERROR) << "Erase eager free buf : " << mem_buf->ToJson() << " failed.";
    }
    mem_stat_ptr_->eager_free_size_ -= mem_buf->size_;
  }
  void InsertEagerFreeBuf(MemBuf *mem_buf) {
    (void)eager_free_mem_bufs_.emplace(mem_buf);
    mem_stat_ptr_->eager_free_size_ += mem_buf->size_;
  }

  /// @brief Merge two mem bufs to one mem buf, merge from src to dst
  ///
  /// @param src The mem buf that will be deleted after merge
  /// @param dst The mem buf that will be kept after merge
  void MergeMemBuf(MemBuf *src, MemBuf *dst);

  friend AbstractDynamicMemPool;
};
using MemBufAllocatorPtr = std::shared_ptr<MemBufAllocator>;

using Lock = memory::mem_pool::Lock;
using LockGuard = memory::mem_pool::LockGuard;
class BACKEND_EXPORT AbstractDynamicMemPool : virtual public DynamicMemPool {
 public:
  AbstractDynamicMemPool();
  ~AbstractDynamicMemPool() override = default;

  void Initialize(size_t init_size, size_t increase_size, size_t max_size) override;

  void ReleaseDeviceRes() override;

  // The main program entry of memory alloc.
  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                              uint32_t stream_id = kDefaultStreamIndex) override;

  // Alloc mem buf from mem pool, return mem buf and its allocator
  std::pair<MemBuf *, MemBufAllocator *> AllocMemBuf(size_t align_size, bool from_persistent_mem = false,
                                                     uint32_t stream_id = kDefaultStreamIndex);

  // The main program entry of continuous memory alloc.
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex) override;
  // The main program entry of memory free.
  void FreeTensorMem(const DeviceMemPtr &device_addr) override;

  /// \brief Check if the memory is not event used
  /// \param[in] device_addr The device memory address to check
  /// \return bool True when no event bind on device address
  bool IsNotEventUsedTensorMem(const DeviceMemPtr &device_addr) override;

  bool DoFreeTensorMem(const DeviceMemPtr &device_addr) override;
  // The main program entry of part memory free and part memory keep.
  void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs, const std::vector<DeviceMemPtr> &keep_addrs,
                          const std::vector<size_t> &keep_addr_sizes) override;
  virtual std::vector<MemBuf *> DoFreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                     const std::vector<DeviceMemPtr> &keep_addrs,
                                                     const std::vector<size_t> &keep_addr_sizes);

  // Element in vector : memory_stream_id, address
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &event) override;
  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override;
  bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) override;
  bool SyncAllEvents() override;
  bool DoSyncAllEvents();

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override;
  void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize) override {
    common_unit_size_ = common_size;
    persist_unit_size_ = persist_size;
  }
  size_t MemAllocUnitSize(bool from_persistent_mem = false) const override {
    return from_persistent_mem ? persist_unit_size_ : common_unit_size_;
  }

  void DefragMemory() override;

  void DumpDynamicMemPoolStateInfo() override;
  std::string DynamicMemPoolStateInfo() const;
  void DumpDynamicMemPoolDebugInfo() override;

  // The statistics information.
  size_t TotalMemStatistics() const override;
  size_t TotalUsedMemStatistics() const override;
  size_t TotalUsedByEventMemStatistics() const override;
  size_t TotalIdleMemStatistics() const override;
  size_t TotalEagerFreeMemStatistics() const override;
  size_t UsedMemPeakStatistics() const override;
  size_t MaxMemAllocatedStatistics() const override;
  size_t MaxMemReservedStatistics() const override;
  size_t ActualPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
    const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
    const override;
  void ResetMaxMemReserved() override;
  void ResetMaxMemAllocated() override;

  const bool IsEnableVmm() const override { return enable_vmm_; }

  void SetEnableVmm(bool enable_vmm) override { enable_vmm_ = enable_vmm; }

  // Get method for proxy.
  std::unordered_map<void *, std::pair<MemBuf *, MemBufAllocator *>> &addr_mem_buf_allocators() {
    return addr_mem_buf_allocators_;
  }

  std::unordered_map<std::pair<uint32_t, uint32_t>, std::set<MemBuf *>, pair_hash> &stream_pair_mem_bufs() {
    return stream_pair_mem_bufs_;
  }

  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() override;

  size_t ReleaseFreeBlocks() override;
  size_t ReleaseCustomFreeBlocks();

  const MemStatPtr &mem_stat_ptr() const { return mem_stat_ptr_; }

  Lock &lock() { return lock_; }

  /// @brief Check whether to use a small memory pool.
  ///
  /// Since the persistent memory pool does not release memory frequently,
  /// using a small memory pool will not bring significant benefits. Use small
  /// pool only if the allocate size is less than kSmallSize and the memory pool
  /// is not persistent.
  ///
  /// @param size Size to allocate.
  /// @param is_persistent True if the memory is persistent, false otherwise.
  /// @return True if the size is small enough to use small pool, false otherwise.
  bool UseSmallPool(size_t size, bool is_persistent) {
    if (!IsEnableSmallPool()) {
      return false;
    }
    return is_persistent ? false : size <= kSmallSize;
  }

 protected:
  void WaitPipelineHelper();

  MemBufAllocatorPtr GenerateAllocator(const AllocatorInfo &allocator_key);
  MemBufAllocator *GetMemBufAllocator(size_t size, bool from_persistent_mem, uint32_t stream_id);
  virtual MemBufAllocatorPtr GenerateCustomAllocator(uint32_t stream_id) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }
#ifndef ENABLE_TEST

 protected:
#else

 public:
#endif
  std::map<AllocatorInfo, MemBufAllocatorPtr> stream_id_allocators_;
  std::unordered_map<void *, std::pair<MemBuf *, MemBufAllocator *>> addr_mem_buf_allocators_;
  std::unordered_map<std::pair<uint32_t, uint32_t>, std::set<MemBuf *>, pair_hash> stream_pair_mem_bufs_;
  std::map<uint32_t, MemBufAllocatorPtr> customized_allocators_;
  MemStatPtr mem_stat_ptr_{std::make_shared<MemStat>()};

  bool enable_vmm_{false};
  bool enable_custom_allocator_{false};
  std::function<MallocFuncType> custom_alloc_fn_;
  std::function<FreeFuncType> custom_free_fn_;
  size_t common_unit_size_{kDynamicMemAllocUnitSize};
  size_t persist_unit_size_{kDynamicMemAllocUnitSize};

  size_t eager_free_count_{0};
  size_t last_eager_free_count_{0};
  Lock lock_;

  // init_size_ is for persistent and common.
  size_t init_size_{kDynamicMemAllocUnitSize};
  size_t increase_size_{kDynamicMemAllocUnitSize};
  // Not enable currently.
  size_t max_size_{0};

  bool enable_dump_memory_{false};
};

class BACKEND_EXPORT AbstractEnhancedDynamicMemPool : public AbstractDynamicMemPool {
 public:
  AbstractEnhancedDynamicMemPool();
  AbstractEnhancedDynamicMemPool(const AbstractEnhancedDynamicMemPool &) = delete;
  AbstractEnhancedDynamicMemPool &operator=(const AbstractEnhancedDynamicMemPool &) = delete;
  ~AbstractEnhancedDynamicMemPool() override = default;

  // Report memory pool stat info for enhanced processing.
  virtual void ReportMemoryPoolInfo();
  // Report memory pool stat info for mstx
  virtual void ReportMemoryPoolMallocInfoToMstx(void *ptr, size_t size);
  virtual void ReportMemoryPoolFreeInfoToMstx(void *ptr);
  bool IsEnableTimeEvent() override { return enable_time_event_; }

  void SetEnableTimeEvent(bool enable_time_event) override { enable_time_event_ = enable_time_event; }

  virtual MemoryTimeEventPtr GenAllocateMemoryTimeEvent(const void *addr, size_t size, uint32_t stream_id,
                                                        bool from_persistent, bool is_persistent);

  virtual MemoryTimeEventPtr GenFreeMemoryTimeEvent(const void *addr);

 private:
  std::atomic<bool> enable_time_event_{false};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_ABSTRACT_DYNAMIC_MEM_POOL_H_
