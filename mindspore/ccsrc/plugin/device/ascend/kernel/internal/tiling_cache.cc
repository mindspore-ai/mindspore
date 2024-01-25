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
#include <plugin/device/ascend/kernel/internal/tiling_cache.h>
#include <set>
#include "ops/op_utils.h"

namespace mindspore::kernel {

template <typename... Args>
std::string TilingCacheMgr::GenTilingCacheKey(const std::string &name, const Args &... args) {
  ResetCacheKey();
  GenCache(name, args...);
  std::string str(buf_);
  return str;
}

std::string TilingCacheMgr::GenTilingCacheKey(const std::string &name, PrimitivePtr prim,
                                              const std::vector<KernelTensor *> &inputs) {
  ResetCacheKey();
  ConcatKey(name.c_str(), static_cast<int64_t>(name.size()));
  std::set<int64_t> value_depend_list = ops::GetInputDependValueList(prim);
  for (size_t i = 0; i < inputs.size(); i++) {
    GenCache(inputs[i]);
    if (value_depend_list.find(i) != value_depend_list.end()) {
      // All the depend value has host value.
      ConcatKey(inputs[i]->GetHostData()->addr, inputs[i]->GetHostData()->size);
    }
  }
  std::string str(buf_);
  return str;
}

TilingInfo TilingCacheMgr::GetOrCreateTilingInfo(
  const std::string &key, const std::function<int(internal::HostRawBuf &, internal::RunInfo &)> &tiling_func,
  size_t tiling_size) {
  // Check in cache_buf_
  auto iter = cache_buf_.find(key);
  if (iter != cache_buf_.end()) {
    return iter->second;
  }
  // Need free the dev mem after launch when the cache is full.
  FreeMemoryIfFull();

  // Host addr will be free in tiling_func.
  void *host_addr = malloc(tiling_size);
  TilingInfo tiling_cache_elem;
  host_tiling_buf_.addr_ = host_addr;
  host_tiling_buf_.size_ = tiling_size;

  bool ret = tiling_func(host_tiling_buf_, tiling_cache_elem.run_info_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Tiling failed!";
  }
  // Allocate device tiling buf_
  SetDevAddr(&tiling_cache_elem, tiling_size);

  ret = aclrtMemcpy(tiling_cache_elem.device_buf_.addr_, tiling_cache_elem.device_buf_.size_, host_tiling_buf_.addr_,
                    host_tiling_buf_.size_, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "ACL_MEMCPY_HOST_TO_DEVICE failed!";
  }
  AppendToCache(key, tiling_cache_elem);
  return tiling_cache_elem;
}

void TilingCacheMgr::Clear() {
  cache_buf_.clear();
  // The device memory is allocated by mempool, it will release in destruction.
}
}  // namespace mindspore::kernel
