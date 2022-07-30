/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_EMBEDDING_CACHE_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_EMBEDDING_CACHE_MANAGER_H_
#include <memory>
#include <map>
#include <string>
#include <vector>
#include "include/api/kernel.h"
#include "include/api/status.h"
#include "include/api/data_type.h"
#include "src/litert/delegate/parameter_cache/embedding_cache.h"
#include "src/litert/delegate/parameter_cache/load_host_cache_model.h"
#include "src/litert/delegate/tensorrt/distribution/distribution_base.h"

namespace mindspore {
namespace cache {
class EmbeddingCacheManager {
 public:
  EmbeddingCacheManager() {
    rank_id_ = lite::GetRankID();
    rank_group_size_ = lite::GetGPUGroupSize();
  }
  Status Init(const std::string &cache_model_path, size_t vocab_size, size_t device_cache_size);
  Status Init(DelegateModel<schema::Primitive> *model, size_t vocab_size, size_t device_cache_size);
  bool CheckIsCacheKernel(kernel::Kernel *kernel);
  Status InitCacheKernel(kernel::Kernel *kernel, uint32_t device_id, const void *context);
  bool IsCacheTensor(mindspore::MSTensor tensor);
  int CacheHandle(const std::string &tensor_name, mindspore::MSTensor model_input_tensor, void *device_addr);
  Status SetDeviceCacheAddr(const std::string &tensor_name, void *device_mem_addr, size_t size);
  std::vector<int64_t> GetCacheShape(mindspore::MSTensor tensor);
  size_t GetCacheDataSize(mindspore::MSTensor tensor);

 private:
  std::map<std::string, std::shared_ptr<EmbeddingCache>> caches_;
  std::vector<int> hash_indices_;
  int rank_id_{0};
  int rank_group_size_{1};

  std::shared_ptr<HostCacheModel> host_cache_model_;
  size_t vocab_size_;
  size_t device_cache_size_;
};
}  // namespace cache
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_EMBEDDING_CACHE_MANAGER_H_
