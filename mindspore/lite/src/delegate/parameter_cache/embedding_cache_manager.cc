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
#include "src/delegate/parameter_cache/embedding_cache_manager.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace cache {
Status EmbeddingCacheManager::Init(const std::string &cache_model_path, size_t vocab_size) {
  if (cache_model_path.empty()) {
    MS_LOG(INFO) << "no cache model ";
    return kSuccess;
  }

  host_cache_model_ = std::make_shared<HostCacheModel>();
  if (host_cache_model_ == nullptr) {
    MS_LOG(ERROR) << "HostCacheModel malloc failed";
    return kLiteMemoryFailed;
  }
  auto ret = host_cache_model_->LoadCache(cache_model_path);
  vocab_size_ = vocab_size;
  MS_LOG(INFO) << "cache manager init end, ret " << ret.ToString();
  return ret;
}

bool EmbeddingCacheManager::CheckIsCacheKernel(kernel::Kernel *kernel) {
  if (host_cache_model_ == nullptr) {
    return false;
  }
  return host_cache_model_->CheckIsCacheKernel(kernel);
}

Status EmbeddingCacheManager::InitCacheKernel(kernel::Kernel *kernel, uint32_t device_id, const void *context) {
  if (host_cache_model_ == nullptr) {
    MS_LOG(ERROR) << "cache model is nullptr, kernel " << kernel->name() << " init cache failed";
    return kLiteError;
  }
  auto host_cache_tensor = host_cache_model_->GetHostCacheTensor(kernel);
  if (host_cache_tensor == nullptr) {
    MS_LOG(ERROR) << kernel->name() << ": invalid cache kernel";
    return kLiteError;
  }

  // only support embedding cache
  if (kernel->type() != schema::PrimitiveType_Gather) {
    MS_LOG(ERROR) << kernel->name() << " is not embedding kernel";
    return kLiteError;
  }

  auto tensor = kernel->inputs()[0];
  if (tensor.Shape()[1] != host_cache_tensor.Shape()[1]) {
    MS_LOG(ERROR) << kernel->name() << " embedding_size is invalid, device size is " << tensor.Shape()[1]
                  << " host size is " << host_cache_tensor.Shape()[1];
    return kLiteError;
  }
  size_t vocab_size = vocab_size_;
  size_t host_cache_size = host_cache_tensor.ElementNum();
  size_t device_cache_size = tensor.Shape()[0];
  size_t embedding_size_ = tensor.Shape()[1];
  DataType data_type = tensor.DataType();
  size_t batch_elements = kernel->inputs()[1].ElementNum();
  auto cache =
    std::make_shared<EmbeddingCache>(vocab_size, host_cache_size, device_cache_size, embedding_size_, batch_elements,
                                     data_type, host_cache_tensor.MutableData(), rank_id_, rank_group_size_);
  if (cache == nullptr) {
    MS_LOG(ERROR) << kernel->name() << ": malloc EmbeddingCache failed";
    return kLiteError;
  }

  auto ret = cache->Init(device_id, context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << kernel->name() << ": EmbeddingCache init failed";
    return kLiteError;
  }
  caches_[tensor.Name()] = cache;

  MS_LOG(INFO) << kernel->name() << " is cache kernel, input tensor " << kernel->inputs()[1].Name() << ", cache tensor "
               << tensor.Name();

  return kSuccess;
}

bool EmbeddingCacheManager::IsCacheTensor(mindspore::MSTensor tensor) {
  if (host_cache_model_ == nullptr) {
    return false;
  }
  auto cache = caches_.find(tensor.Name());
  if (cache != caches_.end()) {
    return true;
  }
  return false;
}

Status EmbeddingCacheManager::SetDeviceCacheAddr(const std::string &tensor_name, void *device_mem_addr, size_t size) {
  auto cache_iter = caches_.find(tensor_name);
  if (cache_iter == caches_.end() || cache_iter->second == nullptr) {
    MS_LOG(ERROR) << "not find cache, " << tensor_name;
    return kLiteError;
  }
  auto cache = cache_iter->second;
  return cache->SetDeviceCacheAddr(device_mem_addr, size);
}

// device_addr is model input device addr
int EmbeddingCacheManager::CacheHandle(const std::string &tensor_name, mindspore::MSTensor model_input_tensor,
                                       void *model_input_device_addr) {
  auto cache_iter = caches_.find(tensor_name);
  if (cache_iter == caches_.end()) {
    MS_LOG(ERROR) << "not find cache, " << tensor_name;
    return lite::RET_ERROR;
  }
  auto cache = cache_iter->second;
  hash_indices_.resize(model_input_tensor.ElementNum());
  auto ret = cache->CheckCacheHit(static_cast<int *>(model_input_tensor.MutableData()), hash_indices_.size(),
                                  hash_indices_.data());
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "CheckCacheHit failed, " << model_input_tensor.Name();
    return lite::RET_ERROR;
  }

  for (size_t i = 0; i < hash_indices_.size(); i++) {
    if (hash_indices_[i] != -1) {
      hash_indices_[i] += cache->GetDeviceStartIndex();
    }
  }

  auto cuda_ret = cudaMemcpy(model_input_device_addr, hash_indices_.data(), hash_indices_.size() * sizeof(int),
                             cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "copy mem failed, " << model_input_tensor.Name();
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "cache handle succ, " << model_input_tensor.Name() << "," << tensor_name;

  return lite::RET_OK;
}
}  // namespace cache
}  // namespace mindspore
