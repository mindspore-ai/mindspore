/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/ascend_gmem_adapter.h"
#include <pthread.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <tuple>
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace device {
namespace ascend {
static constexpr const char kGMemLibName[] = "libgmem.so";
static constexpr const char kMsEnableGmem[] = "MS_ENABLE_GMEM";
constexpr uint64_t kAscendMmapAlignSize = 1 << 21;
constexpr int kMapPeerShared = 0x8000000;

void *callback_thread_func(void *data);

struct CallbackThread {
  ~CallbackThread() { cancel(); }

  // pthread_cancel may cause bug now, so just set flag to false.
  void cancel() {
    if (flag_.load()) {
      flag_.store(false);
    }
  }

  int create() {
    flag_.store(true);
    return pthread_create(&thread_, nullptr, &callback_thread_func, this);
  }

  pthread_t thread_;
  std::atomic_bool flag_{true};
  int32_t default_timeout_{100};
};

void *callback_thread_func(void *data) {
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);
#ifdef WITH_BACKEND
  auto callback_thread = reinterpret_cast<CallbackThread *>(data);
  while (callback_thread->flag_.load()) {
    aclrtProcessReport(callback_thread->default_timeout_);
  }
  MS_LOG(INFO) << "Exit callback thread loop.";
#endif
  return data;
}

const size_t AscendGmemAdapter::GetRoundUpAlignSize(size_t input_size) const {
  return (input_size + kAscendMmapAlignSize - 1) & ~(kAscendMmapAlignSize - 1);
}

const size_t AscendGmemAdapter::GetRoundDownAlignSize(size_t input_size) const {
  return input_size & ~(kAscendMmapAlignSize - 1);
}

size_t AscendGmemAdapter::AllocDeviceMem(size_t size, DeviceMemPtr *addr) const {
  size_t align_size = GetRoundUpAlignSize(size);
  uint8_t *alloc_addr = MmapMemory(align_size, nullptr);
  if (alloc_addr == nullptr) {
    MS_LOG(WARNING) << "Malloc memory failed.";
    return 0;
  }
  *addr = alloc_addr;
  return align_size;
}

size_t AscendGmemAdapter::EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) const {
  MS_LOG(DEBUG) << "Enter ascend eager free device mem, addr : " << addr << ", size : " << size << ".";
  if (size <= 0) {
    MS_LOG(WARNING) << "Size is non positive.";
    return 0;
  }
  size_t addr_size_t = reinterpret_cast<size_t>(addr);
  // Adjust addr -> round up addr, size -> round down size.
  size_t from_addr = GetRoundUpAlignSize(addr_size_t);
  size_t end_addr = GetRoundDownAlignSize(addr_size_t + size);
  if (end_addr <= from_addr) {
    MS_LOG(DEBUG) << "End addr : " << end_addr << " is not bigger than from_addr : " << from_addr << ".";
    return 0;
  }
  size_t real_size = end_addr - from_addr;
  int ret = free_eager_(from_addr, SizeToUlong(real_size), nullptr);
  return ret != 0 ? 0 : real_size;
}

bool AscendGmemAdapter::AddCallbackThread(rtStream_t stream) {
  MS_LOG(DEBUG) << "Enter add callback thread, stream : " << stream << ".";
  if (!is_eager_free_enabled()) {
    return false;
  }
#ifdef WITH_BACKEND
  if (callback_map_.count(stream) > 0) {
    MS_LOG(WARNING) << "Add callback thread failed, stream : " << stream << " is already added.";
    return false;
  }

  auto callback_thread = std::make_shared<CallbackThread>();
  callback_thread->create();
  auto ret = aclrtSubscribeReport(callback_thread->thread_, (aclrtStream)stream);
  if (!ret) {
    MS_LOG(INFO) << "Add callback thread success, stream : " << stream << ".";
    (void)callback_map_.emplace(stream, callback_thread);
    return true;
  } else {
    MS_LOG(ERROR) << "Add callback thread failed, stream : " << stream << ", ret : " << ret;
  }
#endif
  return false;
}

bool AscendGmemAdapter::RemoveCallbackThread(rtStream_t stream) {
  MS_LOG(DEBUG) << "Enter remove callback thread, stream : " << stream << ".";
  if (!is_eager_free_enabled()) {
    return false;
  }
#ifdef WITH_BACKEND
  if (callback_map_.count(stream) == 0) {
    MS_LOG(WARNING) << "Remove callback thread failed, stream : " << stream << " is not exist.";
    return false;
  }
  auto callback_thread = callback_map_.at(stream);
  // Cannot call aclrtUnSubscribeReport.
  callback_thread->cancel();
  callback_map_.erase(stream);
#endif
  return true;
}

uint8_t *AscendGmemAdapter::MmapMemory(size_t size, void *addr) const {
  MS_LOG(DEBUG) << "Enter mmap memory, size : " << size << ".";
  if (size <= 0) {
    MS_LOG(ERROR) << "Size : " << size << " is non positive.";
    return nullptr;
  }

  int flags = MAP_PRIVATE | MAP_ANONYMOUS | kMapPeerShared;
  int prot = PROT_READ | PROT_WRITE;
  void *mapped_addr = mmap(addr, size, prot, flags, -1, 0);
  if (mapped_addr == MAP_FAILED) {
    MS_LOG(EXCEPTION) << "Mmap failed.";
  }
  return static_cast<uint8_t *>(mapped_addr);
}

bool AscendGmemAdapter::MunmapMemory(void *addr, const size_t size) const {
  MS_LOG(DEBUG) << "Enter munmap memory, addr : " << addr << ", size : " << size << ".";
  auto ret = munmap(addr, size);
  return ret != -1;
}

void AscendGmemAdapter::LoadGMemLib() noexcept {
  if (common::GetEnv(kMsEnableGmem) != "1") {
    return;
  }
  MS_LOG(INFO) << "MS_ENABLE_GMEM is set, try to open gmem.";
  gmem_handle_ = dlopen(kGMemLibName, RTLD_NOW);
  if (gmem_handle_ != nullptr) {
    MS_LOG(WARNING) << "Open GMem lib success, mindspore will use gmem to optimize memory usage.";
    LIB_FUNC(GMEM_FREE_EAGER) gmem_free_eager = DlsymFuncObj(gmemFreeEager, gmem_handle_);
    if (gmem_free_eager != nullptr) {
      is_eager_free_enabled_ = true;
      free_eager_ = gmem_free_eager;
    } else {
      MS_LOG(WARNING) << "Load gmem free eager failed.";
      if (dlclose(gmem_handle_) != 0) {
        MS_LOG(ERROR) << "Close GMem lib failed, detail : " << dlerror() << ".";
      }
    }
  } else {
    MS_LOG(INFO) << "Open GMem lib failed.";
  }
}

void AscendGmemAdapter::UnloadGMemLib() noexcept {
  if (gmem_handle_ != nullptr) {
    MS_LOG(INFO) << "Close GMem lib.";
    if (dlclose(gmem_handle_) != 0) {
      MS_LOG(ERROR) << "Close GMem lib failed, detail : " << dlerror() << ".";
    }
    gmem_handle_ = nullptr;
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
