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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ACME_OP_CACHE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ACME_OP_CACHE_H_

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "mindspore/core/ir/primitive.h"
#include "kernel/kernel.h"
#include "acme/include/acme.h"
#include "plugin/device/ascend/kernel/internal/acme/tiling_mem_mgr.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMaxKernelCount = kTilingMemPoolDeviceBlockNum;

struct TilingCacheItem {
  std::atomic<int64_t> ref_count_{0};
  acme::TilingInfoPtr tiling_info_;
  void *host_addr_;
  size_t size_;

  TilingCacheItem(const acme::TilingInfoPtr &tiling_info, void *host_addr, size_t size)
      : ref_count_(1), tiling_info_(tiling_info), host_addr_(host_addr), size_(size) {}
};
using TilingCacheItemPtr = std::shared_ptr<TilingCacheItem>;

class AcmeTilingCache {
 public:
  AcmeTilingCache() = default;
  ~AcmeTilingCache() = default;

  static AcmeTilingCache &GetInstance() {
    static AcmeTilingCache tiling_cache;
    return tiling_cache;
  }

  TilingCacheItemPtr Bind(uint64_t key);
  void Unbind(const TilingCacheItemPtr &item);
  bool Insert(uint64_t key, const TilingCacheItemPtr &ti_ptr);
  std::vector<TilingCacheItemPtr> CombOutSuspectedUselessItems();

  static uint64_t GenerateKey(const std::string &name, const std::vector<KernelTensor *> &inputs);

 private:
  std::unordered_map<uint64_t, TilingCacheItemPtr> cache_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ACME_OP_CACHE_H_
