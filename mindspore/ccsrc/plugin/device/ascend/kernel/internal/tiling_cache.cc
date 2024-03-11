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
#include "plugin/device/ascend/kernel/internal/tiling_cache.h"

#include <set>

#include "ops/op_utils.h"
#include "transform/acl_ir/op_api_cache.h"

namespace mindspore::kernel {

uint64_t TilingCacheMgr::GenTilingCacheKey(const std::string &name, PrimitivePtr prim,
                                           const std::vector<KernelTensor *> &inputs) {
  std::lock_guard<std::mutex> lock(key_mtx_);
  ResetCacheKey();
  ConcatKey(name.c_str(), static_cast<int64_t>(name.size()));
  std::set<int64_t> value_depend_list = ops::GetInputDependValueList(prim);
  for (size_t i = 0; i < inputs.size(); i++) {
    if (value_depend_list.find(i) != value_depend_list.end()) {
      // Should cache the value of depend inputs
      ConcatKey(inputs[i]->GetValuePtr(), static_cast<int64_t>(inputs[i]->size()));
    } else {
      GenCache(inputs[i]);
    }
  }
  return calc_hash_id();
}

TilingInfo TilingCacheMgr::GetOrCreateTilingInfo(
  const uint64_t key, const std::function<int(internal::HostRawBuf &, internal::CacheInfo &)> &tiling_func,
  size_t tiling_size) {
  std::lock_guard<std::mutex> lock(cache_mtx_);
  // Check in cache_buf_
  auto iter = cache_buf_.find(key);
  if (iter != cache_buf_.end() && key != 0) {
    return iter->second;
  }
  // Need free the dev mem after launch when the cache is full.
  FreeMemoryIfFull();

  // Malloc host addr for tiling_func.
  void *host_addr = malloc(tiling_size);
  TilingInfo tiling_cache_elem;
  host_tiling_buf_.addr_ = host_addr;
  host_tiling_buf_.size_ = tiling_size;

  bool ret = tiling_func(host_tiling_buf_, tiling_cache_elem.cache_info_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Tiling failed!";
  }
  if (iter != cache_buf_.end()) {
    tiling_cache_elem = iter->second;
  } else {
    if (key == 0) {
      dev_offset_ = kMaxDevBlockSize;
      tiling_size = kMaxDevBlockSize;
    }
    // Allocate device tiling buf_
    SetDevAddr(&tiling_cache_elem, tiling_size);
  }
  // Bind device to current thread.
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);

  ret = aclrtMemcpy(tiling_cache_elem.device_buf_.addr_, tiling_cache_elem.device_buf_.size_, host_tiling_buf_.addr_,
                    host_tiling_buf_.size_, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "ACL_MEMCPY_HOST_TO_DEVICE failed!";
  }
  free(host_addr);
  AppendToCache(key, tiling_cache_elem);
  return tiling_cache_elem;
}

uint64_t TilingCacheMgr::calc_hash_id() {
  if (offset_ == kBufMaxSize) {
    return 0;
  }
  uint64_t has_id = transform::gen_hash(buf_, offset_);
  return has_id;
}

void TilingCacheMgr::Clear() {
  cache_buf_.clear();
  // The device memory is allocated by mempool, it will release in destruction.
}
}  // namespace mindspore::kernel
